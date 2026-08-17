/* =========================================================================
 * vulkan/vulkan_backend.cpp - Combined Vulkan backend for DS4
 *
 * Part 1: Infrastructure (init, device, memory, commands, model loading)
 * Part 2: GPU function stubs (all ds4_gpu_* functions, CPU fallback)
 *
 * Replace individual stubs with actual Vulkan compute dispatches as
 * GLSL shaders are written.
 * ========================================================================= */

/* Direct Vulkan headers (no volk - we link against libvulkan.so directly) */
#include <vulkan/vulkan.h>

/* VMA (Vulkan Memory Allocator) - use static Vulkan functions from libvulkan.so */
#define VMA_IMPLEMENTATION
#include "include/vk_mem_alloc.h"

#include "../ds4_gpu.h"
#include "../ds4_vulkan.h"
#include "q8_aligned_artifact.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <map>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <algorithm>
#include <chrono>

/* =====================================================================
 * PART 1: Vulkan Device State & Infrastructure
 * ===================================================================== */

struct RoutedTimestamp {
    const char *stage = nullptr;
    uint32_t first_query = 0;
};

enum class TimelineEventKind : uint8_t {
    Dispatch,
    Barrier,
    Submit,
    Wait,
    DescriptorAlloc,
    DescriptorFree,
    TensorAlloc,
    TensorFree,
    BufferAlloc,
    BufferFree,
    WeightUse,
    HostFlush,
    HostInvalidate,
};

struct TimelineEvent {
    TimelineEventKind kind = TimelineEventKind::Dispatch;
    const char *name = nullptr;
    const char *stage = nullptr;
    uint64_t host_ns = 0;
    uint64_t host_end_ns = 0;
    uint64_t duration_ns = 0;
    uint64_t gpu_ns = 0;
    uint64_t gpu_start_ns = 0;
    uint64_t gpu_end_ns = 0;
    uint64_t bytes = 0;
    uint64_t offset = 0;
    uint64_t generation = 0;
    uint32_t x = 0, y = 0, z = 0;
    uint32_t first_query = UINT32_MAX;
    uint32_t count = 0;
};

static constexpr uint32_t DS4_VK_COMMAND_RING_SIZE = 4;
static constexpr uint32_t DS4_VK_TIMELINE_QUERY_COUNT = 2048;
static constexpr uint64_t DS4_VK_TIMELINE_MAX_DISPATCHES =
    DS4_VK_TIMELINE_QUERY_COUNT / 2u;

struct VulkanCommandCtx {
    VkCommandPool   pool     = VK_NULL_HANDLE;
    VkCommandBuffer cmd      = VK_NULL_HANDLE;
    VkSemaphore     semaphore = VK_NULL_HANDLE;
    VkQueryPool     timestamp_pool = VK_NULL_HANDLE;
    uint32_t        timestamp_cursor = 0;
    std::vector<RoutedTimestamp> routed_timestamps;
    uint64_t        event_counter = 0;
    uint32_t        command_count = 0;
    bool            recording = false;
    bool            first_cmd = true;
    VkDescriptorSet ds_q8s = VK_NULL_HANDLE;  /* simple shader DS */
    VkDescriptorSet ds_q8c = VK_NULL_HANDLE;  /* complex shader DS */
    VkCommandBuffer cmd_rots[DS4_VK_COMMAND_RING_SIZE] = {};
    VkQueryPool timestamp_pools[DS4_VK_COMMAND_RING_SIZE] = {};
    uint64_t slot_submit_values[DS4_VK_COMMAND_RING_SIZE] = {};
    uint64_t slot_generations[DS4_VK_COMMAND_RING_SIZE] = {};
    uint32_t slot_timestamp_counts[DS4_VK_COMMAND_RING_SIZE] = {};
    std::vector<RoutedTimestamp> slot_routed_timestamps[DS4_VK_COMMAND_RING_SIZE];
    std::vector<VkDescriptorSet> slot_descriptors[DS4_VK_COMMAND_RING_SIZE];
    std::vector<ds4_gpu_tensor *> slot_tensors[DS4_VK_COMMAND_RING_SIZE];
    std::vector<void *> slot_in_place_ptrs[DS4_VK_COMMAND_RING_SIZE];
    uint32_t cmd_rot_idx = 0;
    uint64_t last_submit_value = 0;
    uint64_t completed_value = 0;
    uint64_t recording_generation = 0;
    bool timeline_enabled = false;
    bool routed_profile_enabled = false;
    bool timeline_collecting = false;
    bool timeline_dumped = false;
    bool timeline_queries_pending = false;
    uint64_t timeline_seen_dispatches = 0;
    uint64_t timeline_skip_dispatches = 0;
    uint64_t timeline_max_dispatches = 512;
    uint64_t timeline_captured_dispatches = 0;
    std::vector<TimelineEvent> timeline_events;
    bool attention_output_batch = false;
    std::vector<VkDescriptorSet> attention_output_descriptors;
    std::vector<ds4_gpu_tensor *> attention_output_tensors;
    bool layer_timeline_active = false;
    bool layer_timeline_end_pending = false;
    uint32_t layer_timeline_layer = UINT32_MAX;
    uint64_t layer_timeline_start_ns = 0;
    uint64_t layer_timeline_stop_ns = 0;
    size_t layer_timeline_stage_cursor = 0;
    bool layer_batch_active = false;
    std::vector<VkDescriptorSet> layer_batch_descriptors;
    std::vector<ds4_gpu_tensor *> layer_batch_tensors;
    std::vector<void *> layer_batch_in_place_ptrs;
};

struct ShaderEntry {
    std::string      name;
    VkShaderModule   module   = VK_NULL_HANDLE;
    VkPipelineLayout layout   = VK_NULL_HANDLE;
    VkPipeline       pipeline = VK_NULL_HANDLE;
    VkDescriptorSetLayout desc_layout  = VK_NULL_HANDLE;
    uint32_t         push_size = 0;
    uint32_t         binding_count = 6;
};

/* Tensor header for VkBuffer/VmaAllocation tracking */
struct TensorHeader {
    VkBuffer       buffer = VK_NULL_HANDLE;
    VmaAllocation  allocation = VK_NULL_HANDLE;
    /* The public tensor pointer is also the lookup key used by the legacy
     * Vulkan ABI.  Keep it explicit so a non-host-visible scratch buffer can
     * use a stable 1-byte identity token without pretending that the buffer
     * is CPU mapped. */
    void          *mapped_ptr = nullptr;
    uint64_t       bytes = 0;
    bool           is_managed = false;
    /* Mapped host-coherent allocations do not need VMA cache maintenance.
     * Keep the property on the allocation header so submit/wait can avoid
     * paying for flush/invalidate calls without changing non-coherent
     * behavior. */
    bool           host_coherent = false;
    bool           host_visible = false;
    bool           reusable_scratch = false;
    bool           device_local_scratch = false;
};

/* ds4_gpu_tensor struct definition comes from ds4_gpu_mgpu.h (the header
 * forward-declares it in ds4_gpu.h).  The full layout
 * (ptr/bytes/owner/device_id) must match the shared multi-GPU plumbing. */
#include "../ds4_gpu_mgpu.h"

/* Forward declarations for VK_CHECK_RAW macro */
#define VK_CHECK_RAW(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "ds4: VULKAN error %d at %s:%d\n", _r, __FILE__, __LINE__); return -1; } } while(0)
#define VK_CHECK_BOOL(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "ds4: VULKAN error %d at %s:%d\n", _r, __FILE__, __LINE__); return 0; } } while(0)
#define VK_CHECK_VOID(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "ds4: VULKAN error %d at %s:%d\n", _r, __FILE__, __LINE__); return; } } while(0)

/* Global state */
static struct {
    VkInstance          instance       = VK_NULL_HANDLE;
    VkPhysicalDevice    phys_device    = VK_NULL_HANDLE;
    VkDevice            device         = VK_NULL_HANDLE;
    uint32_t            queue_family   = UINT32_MAX;
    VkQueue             queue          = VK_NULL_HANDLE;
    VmaAllocator        allocator      = VK_NULL_HANDLE;
    ds4_vulkan_caps     caps;
    VkDescriptorPool    desc_pool      = VK_NULL_HANDLE;
    
    std::vector<ShaderEntry> shaders;
    std::unordered_map<std::string, uint32_t> shader_map;
    
    std::recursive_mutex cmd_mutex;
    std::mutex          queue_mutex;
    std::map<std::thread::id, VulkanCommandCtx> cmd_ctxs;
    
    std::unordered_map<void*, TensorHeader*> tensor_headers;
    /* Scratch is deliberately a small, exact-size pool rather than a general
     * allocator.  Entries are returned only from ring retirement (or after a
     * synchronous fallback has completed), so a reused VkBuffer cannot still
     * be referenced by an in-flight command buffer. */
    std::unordered_map<uint64_t, std::vector<TensorHeader*>> scratch_pool;
    std::unordered_map<uint64_t, std::vector<TensorHeader*>> device_scratch_pool;
    ds4_gpu_tensor     routed_iq2_lut;
    
    /* Weight cache: maps model file offset -> VkBuffer with weights copied to
     * GPU.  Ranges are uploaded lazily on first kernel use (see ensure_weight)
     * and evicted LRU-style against g_vk.weight_budget, so models larger than
     * the device heap stream layer-by-layer (llama.cpp-style). */
    struct WeightCacheEntry {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VmaAllocation  allocation = VK_NULL_HANDLE;
        uint64_t       size = 0;
        uint64_t       last_used = 0;
        bool           pinned = false;
        std::vector<uint64_t> active_generations;
        VkDescriptorBufferInfo desc_info{};
    };
    struct AlignedWeightEntry {
        const void *model_map = nullptr;
        uint64_t model_size = 0;
        uint64_t source_offset = 0;
        uint64_t in_dim = 0;
        uint64_t out_dim = 0;
        uint64_t blocks = 0;
        uint64_t scale_bytes = 0;
        uint64_t payload_offset = 0;
        uint64_t payload_bytes = 0;
        WeightCacheEntry gpu;
    };
    std::unordered_map<uint64_t, WeightCacheEntry> weight_cache;
    std::unordered_map<uint64_t, AlignedWeightEntry> aligned_cache; /* source offset -> artifact */
    /* Model tensor ranges registered by cache_model_range (metadata only). */
    std::unordered_map<uint64_t, uint64_t> range_registry; /* offset -> bytes */
    uint64_t weight_budget = 40ull * 1024 * 1024 * 1024;   /* bytes; DS4_VULKAN_WEIGHT_BUDGET_GB overrides */
    uint64_t weight_used = 0;
    uint64_t lru_counter = 0;
    uint64_t cmd_gen = 0;   /* incremented each begin_cmd; guards in-flight eviction */

    /* Single model buffer covering the entire mmap'd model */
    VkBuffer            model_buffer     = VK_NULL_HANDLE;
    VkDeviceMemory      model_mem        = VK_NULL_HANDLE;
    uint8_t            *model_data       = nullptr;

    const void         *model_map      = nullptr;
    uint64_t            model_size     = 0;
    PFN_vkGetMemoryHostPointerPropertiesEXT pfnGetMemoryHostPointerProperties = nullptr;
    bool                quality        = false;
    bool                ssd_streaming  = false;
    uint32_t            expert_cache_budget = 0;
    uint64_t            expert_cache_expert_bytes = 0;
    uint32_t            streamed_experts = 0;
    float               timestamp_period_ns = 0.0f;
    uint32_t            timestamp_valid_bits = 0;
    bool                initialized    = false;
} g_vk;

const char *ds4_vulkan_gpu_name = "unknown";
const char *ds4_vulkan_driver_version = "unknown";

/* ---- Instance / Device Setup ---- */

static bool has_extension(VkPhysicalDevice dev, const char *name) {
    uint32_t count;
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> props(count);
    vkEnumerateDeviceExtensionProperties(dev, nullptr, &count, props.data());
    for (auto &p : props)
        if (strcmp(p.extensionName, name) == 0) return true;
    return false;
}

static int select_physical_device(void) {
    uint32_t count = 0;
    VK_CHECK_RAW(vkEnumeratePhysicalDevices(g_vk.instance, &count, nullptr));
    if (!count) { fprintf(stderr, "ds4: VULKAN no devices\n"); return -1; }
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK_RAW(vkEnumeratePhysicalDevices(g_vk.instance, &count, devices.data()));

    int best_score = -1;
    VkPhysicalDevice best = VK_NULL_HANDLE;
    for (auto dev : devices) {
        VkPhysicalDeviceProperties props;
        VkPhysicalDeviceMemoryProperties mem;
        vkGetPhysicalDeviceProperties(dev, &props);
        vkGetPhysicalDeviceMemoryProperties(dev, &mem);

        int score = 0;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score += 100;
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score += 80;

        uint32_t qcount = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, nullptr);
        std::vector<VkQueueFamilyProperties> qprops(qcount);
        vkGetPhysicalDeviceQueueFamilyProperties(dev, &qcount, qprops.data());
        for (uint32_t i = 0; i < qcount; i++)
            if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { score += 10; break; }

        for (uint32_t i = 0; i < mem.memoryHeapCount; i++)
            if (mem.memoryHeaps[i].size > 64ULL * 1024 * 1024 * 1024) score += 50;

        if (score > best_score) { best_score = score; best = dev; }
    }
    if (!best) { fprintf(stderr, "ds4: VULKAN no suitable device\n"); return -1; }

    g_vk.phys_device = best;
    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(best, &props);
    ds4_vulkan_gpu_name = strdup(props.deviceName);
    char ver[64]; snprintf(ver, 64, "%u.%u.%u",
        VK_VERSION_MAJOR(props.driverVersion),
        VK_VERSION_MINOR(props.driverVersion),
        VK_VERSION_PATCH(props.driverVersion));
    ds4_vulkan_driver_version = strdup(ver);

    /* Check subgroup support. */
    VkPhysicalDeviceSubgroupProperties sg{};
    sg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceProperties2 p2{};
    p2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    p2.pNext = &sg;
    vkGetPhysicalDeviceProperties2(best, &p2);

    g_vk.caps.subgroup_size = sg.subgroupSize;
    g_vk.caps.min_storage_buffer_offset_alignment =
        props.limits.minStorageBufferOffsetAlignment;
    g_vk.caps.max_push_constants_size = props.limits.maxPushConstantsSize;
    g_vk.caps.max_compute_work_group_invocations = props.limits.maxComputeWorkGroupInvocations;
    g_vk.caps.max_shared_memory_size = props.limits.maxComputeSharedMemorySize;
    for (int i = 0; i < 3; ++i) {
        g_vk.caps.max_compute_work_group_count[i] = props.limits.maxComputeWorkGroupCount[i];
        g_vk.caps.max_compute_work_group_size[i] = props.limits.maxComputeWorkGroupSize[i];
    }
    g_vk.caps.max_storage_buffer_range = props.limits.maxStorageBufferRange;

    g_vk.caps.has_subgroup_basic      = !!(sg.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT);
    g_vk.caps.has_subgroup_arithmetic = !!(sg.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT);
    g_vk.caps.has_subgroup_ballot     = !!(sg.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT);
    g_vk.caps.has_subgroup_shuffle    = !!(sg.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT);

    VkPhysicalDeviceMemoryProperties mem;
    vkGetPhysicalDeviceMemoryProperties(best, &mem);
    g_vk.caps.device_memory_total = 0;
    for (uint32_t i = 0; i < mem.memoryHeapCount; i++)
        g_vk.caps.device_memory_total += mem.memoryHeaps[i].size;

    fprintf(stderr, "ds4: VULKAN device: %s driver=%s subgroup=%u max_shmem=%u mem=%lu MB\n",
            props.deviceName, ver, g_vk.caps.subgroup_size, g_vk.caps.max_shared_memory_size,
            (unsigned long)(g_vk.caps.device_memory_total / (1024*1024)));
    return 0;
}

static int create_logical_device(void) {
    uint32_t count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys_device, &count, nullptr);
    std::vector<VkQueueFamilyProperties> qprops(count);
    vkGetPhysicalDeviceQueueFamilyProperties(g_vk.phys_device, &count, qprops.data());

    int qf = -1;
    for (uint32_t i = 0; i < count; i++)
        if (qprops[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    if (qf < 0) { fprintf(stderr, "ds4: VULKAN no compute queue\n"); return -1; }
    g_vk.queue_family = qf;
    g_vk.timestamp_valid_bits = qprops[qf].timestampValidBits;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g_vk.phys_device, &props);
    g_vk.timestamp_period_ns = props.limits.timestampPeriod;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{};
    qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;

    VkPhysicalDeviceFeatures feat{}; feat.shaderInt64 = VK_TRUE;

    VkPhysicalDeviceVulkan11Features f11{};
    f11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;

    VkPhysicalDeviceVulkan12Features f12{};
    f12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f12.shaderFloat16 = VK_TRUE; f12.shaderInt8 = VK_TRUE;
    f12.shaderBufferInt64Atomics = VK_TRUE;
    f12.shaderSharedInt64Atomics = VK_TRUE;
    f12.hostQueryReset = VK_TRUE; f12.timelineSemaphore = VK_TRUE;
    f12.bufferDeviceAddress = VK_TRUE; f12.vulkanMemoryModel = VK_TRUE;
    f11.pNext = &f12;

    VkPhysicalDeviceVulkan13Features f13{};
    f13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f13.maintenance4 = VK_TRUE; f13.shaderDemoteToHelperInvocation = VK_TRUE;
    f13.inlineUniformBlock = VK_TRUE; f13.pipelineCreationCacheControl = VK_TRUE;
    f12.pNext = &f13;

    VkPhysicalDeviceShaderAtomicInt64Features a64{};
    a64.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_INT64_FEATURES;
    a64.shaderBufferInt64Atomics = VK_TRUE; a64.shaderSharedInt64Atomics = VK_TRUE;
    f13.pNext = &a64;

    const char *dext[] = { VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME };
    VkDeviceCreateInfo dci{};
    dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = dext;
    dci.pEnabledFeatures = &feat; dci.pNext = &f11;
    VK_CHECK_RAW(vkCreateDevice(g_vk.phys_device, &dci, nullptr, &g_vk.device));
    vkGetDeviceQueue(g_vk.device, qf, 0, &g_vk.queue);

    VmaAllocatorCreateInfo vaci{};
    vaci.vulkanApiVersion = VK_API_VERSION_1_3;
    vaci.physicalDevice = g_vk.phys_device; vaci.device = g_vk.device;
    vaci.instance = g_vk.instance;
    vaci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    VK_CHECK_RAW(vmaCreateAllocator(&vaci, &g_vk.allocator));

    VkDescriptorPoolSize ps[] = {{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 65536 }};
    VkDescriptorPoolCreateInfo dpci{};
    dpci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    dpci.maxSets = 65536; dpci.poolSizeCount = 1; dpci.pPoolSizes = ps;
    dpci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    VK_CHECK_RAW(vkCreateDescriptorPool(g_vk.device, &dpci, nullptr, &g_vk.desc_pool));
    return 0;
}

/* ---- Shader Compilation ---- */

static VkShaderModule create_shader_module(const uint32_t *spirv, size_t bytes) {
    VkShaderModuleCreateInfo smci{};
    smci.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    smci.codeSize = bytes; smci.pCode = spirv;
    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult r = vkCreateShaderModule(g_vk.device, &smci, nullptr, &mod);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "ds4: VULKAN vkCreateShaderModule failed: %d\n", r);
        return VK_NULL_HANDLE;
    }
    return mod;
}

static int load_spirv(const std::string &path, std::vector<uint32_t> &out) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return -1;
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    if (sz <= 0 || sz % 4) { fclose(f); return -1; }
    out.resize(sz / 4);
    (void)fread(out.data(), 1, sz, f); fclose(f);
    return 0;
}

static int create_compute_pipeline(ShaderEntry &entry) {
    VkDescriptorSetLayoutBinding bindings[9] = {};
    for (uint32_t i = 0; i < entry.binding_count; i++) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dslci{};
    dslci.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    dslci.bindingCount = entry.binding_count; dslci.pBindings = bindings;
    VK_CHECK_RAW(vkCreateDescriptorSetLayout(g_vk.device, &dslci, nullptr, &entry.desc_layout));

    VkPushConstantRange pr{};
    pr.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; pr.offset = 0;
    pr.size = entry.push_size > 0 ? entry.push_size : 128;
    if (pr.size > g_vk.caps.max_push_constants_size)
        pr.size = g_vk.caps.max_push_constants_size;

    VkPipelineLayoutCreateInfo plci{};
    plci.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    plci.setLayoutCount = 1; plci.pSetLayouts = &entry.desc_layout;
    plci.pushConstantRangeCount = 1; plci.pPushConstantRanges = &pr;
    VK_CHECK_RAW(vkCreatePipelineLayout(g_vk.device, &plci, nullptr, &entry.layout));

    VkComputePipelineCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    cpci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    cpci.stage.module = entry.module; cpci.stage.pName = "main";
    cpci.layout = entry.layout;
    VK_CHECK_RAW(vkCreateComputePipelines(g_vk.device, VK_NULL_HANDLE, 1, &cpci, nullptr, &entry.pipeline));
    return 0;
}

static int load_all_shaders(void) {
    struct { const char *name; uint32_t push_size; uint32_t binding_count; } list[] = {
        {"fill_f32", 12, 6}, {"add_f32", 4, 6},
        {"rms_norm", 12, 6}, {"rms_norm_weight", 12, 6},
        {"swiglu", 16, 6}, {"matmul_f32", 12, 6},
        {"matmul_q8_0", 20, 6},  /* 5 x uint32: in_dim, out_dim, n_tok, blocks, y_scale */
        {"matmul_q8_0_aligned", 20, 4},
        {"matmul_q8_0_aligned_bfe", 20, 4},
        {"matmul_q8_0_wave64_bfe", 20, 4},
        {"matmul_q8_0_rows2_bfe", 20, 4},
        {"matmul_q8_0_rows8_bfe", 20, 4},
        {"matmul_q8_0_hc_expand_rows2_bfe", 16, 7},
        {"matmul_q8_0_group_bfe", 20, 4},
        {"matmul_q8_0_group_rows_bfe", 20, 4},
        {"matmul_q8_0_simple", 12, 6}, /* 3 x uint32: in_dim, out_dim, blocks */
        {"quantize_q8_0_prequant", 12, 2},
        {"matmul_q8_0_prequant", 20, 3},
        {"group_copy", 24, 6},
        {"matmul_f16", 12, 6},   /* 3 x uint32 */
        {"matmul_f16_fast", 12, 3}, /* FP64 lane dots with chunked reduction */
        {"rms_norm_weight_rows", 12, 6},
        {"head_rms_norm", 16, 6},  /* n_tok + n_head + head_dim + eps */
        {"rope_tail", 52, 6},      /* 7 x uint32 + 6 x float */
        {"head_rms_norm_rope_tail", 56, 6}, /* 7 x uint32 + 7 x float */
        {"store_raw_kv_f16", 16, 2},
        {"fp8_kv_quantize", 12, 1},
        {"attention_prefill_raw", 16, 4},
        {"attention_decode_mixed", 32, 6},
        {"attention_decode_mixed_wave64", 32, 6},
        {"attention_decode_mixed_rope", 76, 6},
        {"attention_decode_mixed_rope_wave64_512", 76, 6},
        {"attention_mixed_online", 64, 8},
        {"attention_indexed_online_wave64", 108, 8},
        {"attention_decode_raw_batch", 32, 4},
        {"indexer_scores", 32, 4},
        {"indexer_qat", 4, 1},
        {"indexer_topk", 12, 2},
        {"topk_mask", 12, 2},
        {"compressor_store", 32, 5},
        {"compressor_clear", 12, 2},
        {"compressor_set_rows", 32, 5},
        {"compressor_pool", 32, 6},
        {"compressor_pool_state", 8, 3},
        {"compressor_shift_ratio4", 4, 2},
        {"compressor_rope_stride", 56, 1},
        {"hc_weighted_sum", 16, 6}, /* n_embd, n_hc, rows, reserved */
        {"hc_expand", 36, 6}, /* shape, strides, add/split/half flags */
        {"hc_split_weighted_sum", 32, 8}, /* shape, sinkhorn, eps, sum/norm flags */
        {"output_hc_weights", 16, 4}, /* n_hc, n_tokens, eps, reserved */
        {"router_select", 24, 7}, /* n_tokens, hash_rows, token, bias/hash, scale */
        {"routed_moe", 68, 6}, /* canonical Q8_K/IQ2_XXS/Q2_K routed MoE */
        {"routed_moe_mode0", 68, 6}, {"routed_moe_mode1", 68, 6},
        {"routed_moe_mode2", 68, 6}, {"routed_moe_mode3", 68, 6},
        {"routed_moe_mode4", 68, 6}, {"routed_moe_mode5", 68, 6},
        {"routed_moe_fused", 68, 9}, /* fused IQ2 gate/up/SwiGLU */
        {"routed_moe_fused_mid", 68, 7}, /* exact fused IQ2/SwiGLU -> Q8 mid */
        {"routed_moe_fused_mid_wave64", 68, 7}, /* Wave64 fused mid */
        {"routed_moe_down_reduce_q2", 68, 5}, /* exact Flash Q2 down+reduce */
        {"routed_moe_down_reduce_q2_wave64", 68, 5}, /* Wave64 Q2 down */
    };
    for (auto &l : list) {
        std::string path = std::string("vulkan/shaders/spv/") + l.name + ".spv";
        std::vector<uint32_t> spv;
        if (load_spirv(path, spv) != 0) {
            fprintf(stderr, "ds4: VULKAN shader not found: %s\n", path.c_str());
            continue;
        }
        ShaderEntry e;
        e.name = l.name; e.push_size = l.push_size; e.binding_count = l.binding_count;
        e.module = create_shader_module(spv.data(), spv.size() * 4);
        if (create_compute_pipeline(e) != 0) {
            vkDestroyShaderModule(g_vk.device, e.module, nullptr);
            continue;
        }
        g_vk.shader_map[e.name] = g_vk.shaders.size();
        g_vk.shaders.push_back(e);
    }
    fprintf(stderr, "ds4: VULKAN loaded %zu shaders\n", g_vk.shaders.size());
    return 0;
}

/* ---- Command Context ---- */

static VulkanCommandCtx &get_cmd_ctx(void) {
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    auto tid = std::this_thread::get_id();
    auto it = g_vk.cmd_ctxs.find(tid);
    if (it != g_vk.cmd_ctxs.end()) return it->second;

    VulkanCommandCtx ctx;
    VkCommandPoolCreateInfo cpci{};
    cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    cpci.queueFamilyIndex = g_vk.queue_family;
    cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if (vkCreateCommandPool(g_vk.device, &cpci, nullptr, &ctx.pool) != VK_SUCCESS) {
        fprintf(stderr, "ds4: VULKAN failed to create command pool\n");
        abort();
    }
    VkSemaphoreCreateInfo sci{}; sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    VkSemaphoreTypeCreateInfo stci{};
    stci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    stci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    stci.initialValue = 0;
    sci.pNext = &stci;
    if (vkCreateSemaphore(g_vk.device, &sci, nullptr, &ctx.semaphore) != VK_SUCCESS) abort();
    ctx.timeline_enabled = getenv("DS4_VULKAN_TIMELINE") != nullptr;
    ctx.routed_profile_enabled = getenv("DS4_VULKAN_PROFILE_ROUTED_MOE") != nullptr;
    if (const char *skip = getenv("DS4_VULKAN_TIMELINE_SKIP"))
        ctx.timeline_skip_dispatches = strtoull(skip, nullptr, 10);
    if (const char *count = getenv("DS4_VULKAN_TIMELINE_COUNT")) {
        const uint64_t parsed = strtoull(count, nullptr, 10);
        if (parsed != 0)
            ctx.timeline_max_dispatches = std::min(
                parsed, DS4_VK_TIMELINE_MAX_DISPATCHES);
    }
    if (ctx.timeline_enabled)
        ctx.timeline_events.reserve((size_t)ctx.timeline_max_dispatches * 16u + 256u);
    if ((ctx.routed_profile_enabled || ctx.timeline_enabled ||
         getenv("DS4_VULKAN_TIMELINE_LAYER")) &&
        g_vk.timestamp_valid_bits != 0) {
        bool query_pools_ok = true;
        for (uint32_t slot = 0; slot < DS4_VK_COMMAND_RING_SIZE; slot++) {
            VkQueryPoolCreateInfo qpci{};
            qpci.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
            qpci.queryType = VK_QUERY_TYPE_TIMESTAMP;
            qpci.queryCount = DS4_VK_TIMELINE_QUERY_COUNT;
            if (vkCreateQueryPool(g_vk.device, &qpci, nullptr,
                                  &ctx.timestamp_pools[slot]) != VK_SUCCESS) {
                query_pools_ok = false;
                break;
            }
        }
        if (!query_pools_ok) {
            fprintf(stderr, "ds4: VULKAN timestamp query pools unavailable\n");
            for (VkQueryPool &pool : ctx.timestamp_pools) {
                if (pool) vkDestroyQueryPool(g_vk.device, pool, nullptr);
                pool = VK_NULL_HANDLE;
            }
        }
    }
    auto inserted = g_vk.cmd_ctxs.emplace(tid, std::move(ctx));
    return inserted.first->second;
}

static uint64_t timeline_now_ns(void) {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

static const char *timeline_kind_name(TimelineEventKind kind) {
    switch (kind) {
    case TimelineEventKind::Dispatch: return "dispatch";
    case TimelineEventKind::Barrier: return "barrier";
    case TimelineEventKind::Submit: return "submit";
    case TimelineEventKind::Wait: return "wait";
    case TimelineEventKind::DescriptorAlloc: return "descriptor_alloc";
    case TimelineEventKind::DescriptorFree: return "descriptor_free";
    case TimelineEventKind::TensorAlloc: return "tensor_alloc";
    case TimelineEventKind::TensorFree: return "tensor_free";
    case TimelineEventKind::BufferAlloc: return "buffer_alloc";
    case TimelineEventKind::BufferFree: return "buffer_free";
    case TimelineEventKind::WeightUse: return "weight_use";
    case TimelineEventKind::HostFlush: return "host_flush";
    case TimelineEventKind::HostInvalidate: return "host_invalidate";
    }
    return "unknown";
}

static void timeline_json_string(FILE *fp, const char *s) {
    fputc('"', fp);
    if (s) {
        for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
            if (*p == '\\' || *p == '"') fputc('\\', fp);
            if (*p == '\n') { fputs("\\n", fp); continue; }
            if (*p == '\r') { fputs("\\r", fp); continue; }
            if (*p == '\t') { fputs("\\t", fp); continue; }
            if (*p == '\b') { fputs("\\b", fp); continue; }
            if (*p == '\f') { fputs("\\f", fp); continue; }
            if (*p < 0x20) {
                fprintf(fp, "\\u%04x", (unsigned)*p);
                continue;
            }
            fputc(*p, fp);
        }
    }
    fputc('"', fp);
}

static void timeline_write_json(const VulkanCommandCtx &ctx) {
    const char *path = getenv("DS4_VULKAN_TIMELINE_JSON");
    if (!path || !path[0]) return;
    FILE *fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "ds4: VULKAN timeline JSON open failed: %s\n", path);
        return;
    }
    fputs("{\n  \"layer\": ", fp);
    if (ctx.layer_timeline_layer == UINT32_MAX) fputs("null", fp);
    else fprintf(fp, "%u", ctx.layer_timeline_layer);
    fputs(",\n  \"events\": [\n", fp);
    for (size_t i = 0; i < ctx.timeline_events.size(); i++) {
        const TimelineEvent &e = ctx.timeline_events[i];
        fprintf(fp, "    {\"index\":%zu,\"kind\":", i);
        timeline_json_string(fp, timeline_kind_name(e.kind));
        fputs(",\"name\":", fp);
        timeline_json_string(fp, e.name ? e.name : "");
        fputs(",\"stage\":", fp);
        timeline_json_string(fp, e.stage ? e.stage : "unassigned");
        fprintf(fp, ",\"host_ns\":%llu,\"host_end_ns\":%llu,\"duration_ns\":%llu"
                    ",\"gpu_start_ns\":%llu,\"gpu_end_ns\":%llu,\"gpu_ns\":%llu"
                    ",\"generation\":%llu,\"bytes\":%llu,\"offset\":%llu"
                    ",\"x\":%u,\"y\":%u,\"z\":%u,\"count\":%u} %s\n",
                (unsigned long long)e.host_ns, (unsigned long long)e.host_end_ns,
                (unsigned long long)e.duration_ns, (unsigned long long)e.gpu_start_ns,
                (unsigned long long)e.gpu_end_ns, (unsigned long long)e.gpu_ns,
                (unsigned long long)e.generation, (unsigned long long)e.bytes,
                (unsigned long long)e.offset, e.x, e.y, e.z, e.count,
                i + 1 == ctx.timeline_events.size() ? "" : ",");
    }
    fputs("  ]\n}\n", fp);
    fclose(fp);
}

static void timeline_dump(VulkanCommandCtx &ctx) {
    if (!ctx.timeline_enabled || ctx.timeline_dumped || ctx.timeline_events.empty()) return;
    fprintf(stderr,
            "ds4: VULKAN timeline columns=event kind name host_ns host_end_ns duration_ns "
            "gpu_start_ns gpu_end_ns gpu_ns generation bytes offset x y z count\n");
    for (size_t i = 0; i < ctx.timeline_events.size(); i++) {
        const TimelineEvent &event = ctx.timeline_events[i];
        fprintf(stderr, "ds4: VULKAN timeline %zu %s %s %llu %llu %llu %llu %llu %llu %llu %llu %llu %u %u %u %u\n",
                i, timeline_kind_name(event.kind), event.name ? event.name : "-",
                (unsigned long long)event.host_ns,
                (unsigned long long)event.host_end_ns,
                (unsigned long long)event.duration_ns,
                (unsigned long long)event.gpu_start_ns,
                (unsigned long long)event.gpu_end_ns,
                (unsigned long long)event.gpu_ns,
                (unsigned long long)event.generation,
                (unsigned long long)event.bytes,
                (unsigned long long)event.offset,
                event.x, event.y, event.z, event.count);
    }
    timeline_write_json(ctx);
    ctx.timeline_dumped = true;
    ctx.timeline_collecting = false;
}

static void timeline_layer_summary(VulkanCommandCtx &ctx) {
    if (!ctx.layer_timeline_active) return;
    uint64_t gpu_ns = 0;
    uint64_t submit_ns = 0;
    uint64_t fence_ns = 0;
    uint64_t gpu_idle_ns = 0;
    uint64_t host_record_ns = 0;
    uint64_t flush_ns = 0;
    uint64_t invalidate_ns = 0;
    uint64_t descriptor_alloc_ns = 0;
    uint64_t descriptor_free_ns = 0;
    uint64_t last_gpu_end = 0;
    uint32_t gpu_intervals = 0;
    uint32_t submissions = 0;
    uint32_t waits = 0;
    std::unordered_map<std::string, uint64_t> stage_gpu_ns;
    for (const TimelineEvent &event : ctx.timeline_events) {
        if (event.kind == TimelineEventKind::Dispatch) {
            gpu_ns += event.gpu_ns;
            stage_gpu_ns[event.stage ? event.stage : "unassigned"] += event.gpu_ns;
            if (event.gpu_start_ns != 0 && event.gpu_end_ns >= event.gpu_start_ns) {
                if (last_gpu_end != 0 && event.gpu_start_ns > last_gpu_end)
                    gpu_idle_ns += event.gpu_start_ns - last_gpu_end;
                last_gpu_end = std::max(last_gpu_end, event.gpu_end_ns);
                gpu_intervals++;
            }
            if (event.host_end_ns >= event.host_ns)
                host_record_ns += event.host_end_ns - event.host_ns;
        }
        else if (event.kind == TimelineEventKind::Submit) {
            submissions++;
            submit_ns += event.duration_ns;
        } else if (event.kind == TimelineEventKind::Wait) {
            waits++;
            fence_ns += event.duration_ns;
        } else if (event.kind == TimelineEventKind::HostFlush) {
            flush_ns += event.duration_ns;
        } else if (event.kind == TimelineEventKind::HostInvalidate) {
            invalidate_ns += event.duration_ns;
        } else if (event.kind == TimelineEventKind::DescriptorAlloc &&
                   event.name && !strcmp(event.name, "descriptor_cpu")) {
            descriptor_alloc_ns += event.duration_ns;
        } else if (event.kind == TimelineEventKind::DescriptorFree &&
                   event.name && !strcmp(event.name, "descriptor_free_cpu")) {
            descriptor_free_ns += event.duration_ns;
        }
    }
    const uint64_t wall_stop_ns = ctx.layer_timeline_stop_ns != 0
        ? ctx.layer_timeline_stop_ns : timeline_now_ns();
    const uint64_t wall_ns = wall_stop_ns >= ctx.layer_timeline_start_ns
        ? wall_stop_ns - ctx.layer_timeline_start_ns : 0;
    fprintf(stderr,
            "ds4: VULKAN layer_timeline layer=%u wall_ms=%.6f gpu_ms=%.6f "
            "submissions=%u fence_waits=%u submit_cpu_ms=%.6f fence_cpu_ms=%.6f\n",
            ctx.layer_timeline_layer, (double)wall_ns / 1.0e6,
            (double)gpu_ns / 1.0e6, submissions, waits,
            (double)submit_ns / 1.0e6, (double)fence_ns / 1.0e6);
    fprintf(stderr,
            "ds4: VULKAN layer_timeline_detail layer=%u gpu_idle_ms=%.6f "
            "dispatch_gpu_intervals=%u dispatch_record_ms=%.6f "
            "flush_cpu_ms=%.6f invalidate_cpu_ms=%.6f "
            "descriptor_alloc_cpu_ms=%.6f descriptor_free_cpu_ms=%.6f\n",
            ctx.layer_timeline_layer, (double)gpu_idle_ns / 1.0e6,
            gpu_intervals, (double)host_record_ns / 1.0e6,
            (double)flush_ns / 1.0e6, (double)invalidate_ns / 1.0e6,
            (double)descriptor_alloc_ns / 1.0e6,
            (double)descriptor_free_ns / 1.0e6);
    uint64_t attention_ns = 0;
    uint64_t moe_ns = 0;
    uint64_t other_ns = 0;
    for (const auto &[stage, elapsed_ns] : stage_gpu_ns) {
        const bool attention = stage.rfind("attn_", 0) == 0 ||
            stage == "q_path" || stage == "kv_path" ||
            stage.rfind("compressor", 0) == 0 ||
            stage.rfind("indexer_", 0) == 0;
        const bool moe = stage == "router" ||
            stage.rfind("routed_moe", 0) == 0 ||
            stage == "shared_gate_up" || stage == "shared_down";
        if (attention) attention_ns += elapsed_ns;
        else if (moe) moe_ns += elapsed_ns;
        else other_ns += elapsed_ns;
        fprintf(stderr, "ds4: VULKAN layer_stage layer=%u stage=%s gpu_ms=%.6f\n",
                ctx.layer_timeline_layer, stage.c_str(), (double)elapsed_ns / 1.0e6);
    }
    fprintf(stderr,
            "ds4: VULKAN layer_groups layer=%u attention_ms=%.6f moe_ms=%.6f "
            "other_ms=%.6f total_gpu_ms=%.6f\n",
            ctx.layer_timeline_layer, (double)attention_ns / 1.0e6,
            (double)moe_ns / 1.0e6, (double)other_ns / 1.0e6,
            (double)gpu_ns / 1.0e6);
    timeline_write_json(ctx);
    ctx.layer_timeline_active = false;
    ctx.layer_timeline_end_pending = false;
    ctx.layer_timeline_stop_ns = 0;
    ctx.timeline_collecting = false;
    ctx.timeline_enabled = false;
    ctx.timeline_dumped = true;
}

static TimelineEvent *timeline_add(VulkanCommandCtx &ctx, TimelineEventKind kind,
                                   const char *name) {
    if (!ctx.timeline_collecting || ctx.timeline_dumped) return nullptr;
    TimelineEvent event;
    event.kind = kind;
    event.name = name;
    event.host_ns = timeline_now_ns();
    event.generation = ctx.recording_generation;
    ctx.timeline_events.push_back(event);
    return &ctx.timeline_events.back();
}

static void timeline_maybe_start(VulkanCommandCtx &ctx) {
    if (ctx.layer_timeline_end_pending) return;
    if (ctx.timeline_enabled && !ctx.timeline_dumped &&
        ctx.timeline_seen_dispatches >= ctx.timeline_skip_dispatches &&
        ctx.timeline_captured_dispatches < ctx.timeline_max_dispatches)
        ctx.timeline_collecting = true;
}

static void timeline_barrier(VulkanCommandCtx &ctx, const char *name) {
    (void)timeline_add(ctx, TimelineEventKind::Barrier, name);
}

static void timeline_dispatch(VulkanCommandCtx &ctx, const char *name,
                              VkDescriptorBufferInfo *buffers, uint32_t count,
                              uint32_t x, uint32_t y, uint32_t z) {
    const uint64_t dispatch_index = ctx.timeline_seen_dispatches++;
    const bool capture = ctx.timeline_enabled &&
        dispatch_index >= ctx.timeline_skip_dispatches &&
        ctx.timeline_captured_dispatches < ctx.timeline_max_dispatches;
    if (!capture) {
        if (ctx.timeline_collecting && !ctx.timeline_queries_pending &&
            ctx.timeline_captured_dispatches >= ctx.timeline_max_dispatches)
            timeline_dump(ctx);
        vkCmdDispatch(ctx.cmd, x, y, z);
        return;
    }
    ctx.timeline_collecting = true;
    ctx.timeline_queries_pending = true;
    ctx.timeline_captured_dispatches++;
    TimelineEvent *event = timeline_add(ctx, TimelineEventKind::Dispatch, name);
    if (event) {
        event->x = x; event->y = y; event->z = z; event->count = count;
        for (uint32_t i = 0; i < count; i++)
            if (buffers[i].range <= UINT64_MAX - event->bytes)
                event->bytes += buffers[i].range;
        if (ctx.timestamp_pool != VK_NULL_HANDLE &&
            ctx.timestamp_cursor + 1u < DS4_VK_TIMELINE_QUERY_COUNT) {
            event->first_query = ctx.timestamp_cursor;
            ctx.timestamp_cursor += 2;
            vkCmdWriteTimestamp(ctx.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                                ctx.timestamp_pool, event->first_query);
        }
    }
    vkCmdDispatch(ctx.cmd, x, y, z);
    if (event) event->host_end_ns = timeline_now_ns();
    if (event && event->first_query != UINT32_MAX)
        vkCmdWriteTimestamp(ctx.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            ctx.timestamp_pool, event->first_query + 1);
}

static void timeline_descriptors(VulkanCommandCtx &ctx, const char *name,
                                 VkDescriptorBufferInfo *buffers, uint32_t count) {
    if (ctx.timeline_collecting && !ctx.timeline_queries_pending &&
        ctx.timeline_captured_dispatches >= ctx.timeline_max_dispatches) {
        timeline_dump(ctx);
        return;
    }
    timeline_maybe_start(ctx);
    TimelineEvent *event = timeline_add(ctx, TimelineEventKind::DescriptorAlloc, name);
    if (!event) return;
    event->count = count;
    for (uint32_t i = 0; i < count; i++)
        if (buffers[i].range <= UINT64_MAX - event->bytes)
            event->bytes += buffers[i].range;
}

static void timeline_resource(VulkanCommandCtx &ctx, TimelineEventKind kind,
                              const char *name, uint64_t bytes,
                              uint64_t offset = 0) {
    const bool begins_new_work = kind == TimelineEventKind::TensorAlloc ||
        kind == TimelineEventKind::BufferAlloc ||
        kind == TimelineEventKind::WeightUse;
    if (begins_new_work && ctx.timeline_collecting &&
        !ctx.timeline_queries_pending &&
        ctx.timeline_captured_dispatches >= ctx.timeline_max_dispatches) {
        timeline_dump(ctx);
        return;
    }
    timeline_maybe_start(ctx);
    TimelineEvent *event = timeline_add(ctx, kind, name);
    if (event) {
        event->bytes = bytes;
        event->offset = offset;
    }
}

static void timeline_resource_current(TimelineEventKind kind, const char *name,
                                      uint64_t bytes, uint64_t offset = 0) {
    if (!getenv("DS4_VULKAN_TIMELINE")) return;
    timeline_resource(get_cmd_ctx(), kind, name, bytes, offset);
}

static void timeline_duration_current(TimelineEventKind kind, const char *name,
                                      uint64_t start_ns) {
    auto &ctx = get_cmd_ctx();
    if (!getenv("DS4_VULKAN_TIMELINE") && !ctx.layer_timeline_active) return;
    const bool restore_collecting = ctx.timeline_collecting;
    if (!ctx.timeline_collecting && ctx.layer_timeline_end_pending)
        ctx.timeline_collecting = true;
    TimelineEvent *event = timeline_add(ctx, kind, name);
    if (event) {
        event->host_end_ns = timeline_now_ns();
        event->duration_ns = event->host_end_ns - start_ns;
    }
    ctx.timeline_collecting = restore_collecting;
}

static VkResult timeline_device_wait_idle(const char *name) {
    const bool timeline = getenv("DS4_VULKAN_TIMELINE") != nullptr;
    const uint64_t start_ns = timeline_now_ns();
    VkResult result;
    {
        std::lock_guard<std::mutex> lock(g_vk.queue_mutex);
        result = vkDeviceWaitIdle(g_vk.device);
    }
    if (timeline)
        timeline_duration_current(TimelineEventKind::Wait, name, start_ns);
    return result;
}

static void report_slot_timestamps(VulkanCommandCtx &ctx, uint32_t slot) {
    const VkQueryPool pool = ctx.timestamp_pools[slot];
    const uint32_t count = ctx.slot_timestamp_counts[slot];
    if (pool == VK_NULL_HANDLE || count == 0) return;
    std::vector<uint64_t> ticks(count);
    VkResult result = vkGetQueryPoolResults(
        g_vk.device, pool, 0, count,
        (size_t)count * sizeof(uint64_t), ticks.data(), sizeof(uint64_t),
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (result != VK_SUCCESS) {
        fprintf(stderr, "ds4: VULKAN timestamp read failed: %d\n", result);
        ctx.slot_routed_timestamps[slot].clear();
        ctx.slot_timestamp_counts[slot] = 0;
        return;
    }
    const uint64_t mask = g_vk.timestamp_valid_bits >= 64
        ? UINT64_MAX : ((1ull << g_vk.timestamp_valid_bits) - 1ull);
    for (TimelineEvent &event : ctx.timeline_events) {
        if (event.generation != ctx.slot_generations[slot] ||
            event.first_query == UINT32_MAX) continue;
        const uint64_t begin = ticks[event.first_query] & mask;
        const uint64_t end = ticks[event.first_query + 1] & mask;
        event.gpu_start_ns = (uint64_t)((double)begin * g_vk.timestamp_period_ns);
        event.gpu_end_ns = (uint64_t)((double)end * g_vk.timestamp_period_ns);
        event.gpu_ns = (uint64_t)((double)((end - begin) & mask) *
                                  g_vk.timestamp_period_ns);
        event.first_query = UINT32_MAX;
    }
    if (!ctx.slot_routed_timestamps[slot].empty())
        fprintf(stderr, "ds4: VULKAN routed_moe_gpu_ms");
    for (const RoutedTimestamp &timestamp : ctx.slot_routed_timestamps[slot]) {
        const uint64_t begin = ticks[timestamp.first_query] & mask;
        const uint64_t end = ticks[timestamp.first_query + 1] & mask;
        const uint64_t elapsed = (end - begin) & mask;
        const double ms = (double)elapsed * g_vk.timestamp_period_ns / 1.0e6;
        fprintf(stderr, " %s=%.3f", timestamp.stage, ms);
    }
    if (!ctx.slot_routed_timestamps[slot].empty()) fputc('\n', stderr);
    ctx.slot_routed_timestamps[slot].clear();
    ctx.slot_timestamp_counts[slot] = 0;
}

static bool release_tensor_header(TensorHeader *header);

static int retire_slot_resources(VulkanCommandCtx &ctx, uint32_t slot) {
    int ok = 1;
    const uint64_t descriptor_start = timeline_now_ns();
    for (VkDescriptorSet set : ctx.slot_descriptors[slot]) {
        if (vkFreeDescriptorSets(g_vk.device, g_vk.desc_pool, 1, &set) != VK_SUCCESS)
            ok = 0;
    }
    if (!ctx.slot_descriptors[slot].empty())
        timeline_duration_current(TimelineEventKind::DescriptorFree,
                                  "descriptor_free_cpu", descriptor_start);
    for (ds4_gpu_tensor *tensor : ctx.slot_tensors[slot]) {
        if (!tensor) continue;
        if (tensor->owner && tensor->ptr) {
            auto it = g_vk.tensor_headers.find(tensor->ptr);
            if (it != g_vk.tensor_headers.end()) {
                if (getenv("DS4_VULKAN_TIMELINE"))
                    timeline_resource(ctx, TimelineEventKind::TensorFree,
                                      "tensor", it->second->bytes);
                if (!release_tensor_header(it->second))
                    g_vk.tensor_headers.erase(it);
            }
        }
        free(tensor);
    }
    for (void *ptr : ctx.slot_in_place_ptrs[slot]) {
        auto it = g_vk.tensor_headers.find(ptr);
        if (it == g_vk.tensor_headers.end()) continue;
        if (getenv("DS4_VULKAN_TIMELINE"))
            timeline_resource(ctx, TimelineEventKind::TensorFree,
                              "tensor", it->second->bytes);
        if (!release_tensor_header(it->second))
            g_vk.tensor_headers.erase(it);
    }
    ctx.slot_descriptors[slot].clear();
    ctx.slot_tensors[slot].clear();
    ctx.slot_in_place_ptrs[slot].clear();
    return ok;
}

static int retire_completed_slots(VulkanCommandCtx &ctx, uint64_t completed) {
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    int ok = 1;
    if (completed > ctx.completed_value) ctx.completed_value = completed;
    for (uint32_t slot = 0; slot < DS4_VK_COMMAND_RING_SIZE; slot++) {
        const uint64_t value = ctx.slot_submit_values[slot];
        if (value == 0 || value > ctx.completed_value) continue;
        report_slot_timestamps(ctx, slot);
        if (!retire_slot_resources(ctx, slot)) ok = 0;
        ctx.slot_submit_values[slot] = 0;
        ctx.slot_generations[slot] = 0;
    }
    ctx.timeline_queries_pending = false;
    for (uint32_t slot = 0; slot < DS4_VK_COMMAND_RING_SIZE; slot++)
        ctx.timeline_queries_pending |= ctx.slot_timestamp_counts[slot] != 0;
    return ok;
}

static int wait_for_submit_value(VulkanCommandCtx &ctx, uint64_t value,
                                 const char *name) {
    if (value == 0 || value <= ctx.completed_value) return 1;
    uint64_t counter = 0;
    VK_CHECK_BOOL(vkGetSemaphoreCounterValue(g_vk.device, ctx.semaphore, &counter));
    if (counter < value) {
        VkSemaphoreWaitInfo wait{};
        wait.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
        wait.semaphoreCount = 1;
        wait.pSemaphores = &ctx.semaphore;
        wait.pValues = &value;
        const uint64_t wait_start = timeline_now_ns();
        VK_CHECK_BOOL(vkWaitSemaphores(g_vk.device, &wait, UINT64_MAX));
        if (ctx.timeline_collecting) {
            TimelineEvent *event = timeline_add(ctx, TimelineEventKind::Wait, name);
            if (event) event->duration_ns = timeline_now_ns() - wait_start;
        }
        counter = value;
    }
    return retire_completed_slots(ctx, counter);
}

static bool command_ring_enabled(void) {
    const char *env = getenv("DS4_VULKAN_COMMAND_RING");
    return !env || !env[0] || strcmp(env, "0") != 0;
}

static int begin_cmd(void) {
    auto &c = get_cmd_ctx();
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    if (c.recording) return 1;
    if (!command_ring_enabled() &&
        !wait_for_submit_value(c, c.last_submit_value, "begin_cmd_serial"))
        return 0;
    const uint32_t slot = c.cmd_rot_idx;
    if (!wait_for_submit_value(c, c.slot_submit_values[slot], "begin_cmd_slot"))
        return 0;
    c.timestamp_pool = c.timestamp_pools[slot];
    if (c.timestamp_pool != VK_NULL_HANDLE) {
        vkResetQueryPool(g_vk.device, c.timestamp_pool, 0,
                         DS4_VK_TIMELINE_QUERY_COUNT);
        c.timestamp_cursor = 0;
        c.routed_timestamps.clear();
    }
    /* Allocate or reuse CB */
    VkCommandBuffer &cb = c.cmd_rots[slot];
    if (cb == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = c.pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VK_CHECK_BOOL(vkAllocateCommandBuffers(g_vk.device, &cbai, &cb));
    } else {
        VK_CHECK_BOOL(vkResetCommandBuffer(
            cb, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT));
    }
    c.cmd = cb;
    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_BOOL(vkBeginCommandBuffer(c.cmd, &bi));
    c.recording = true;
    c.command_count = 0;
    c.recording_generation = ++g_vk.cmd_gen;
    if (getenv("DS4_VULKAN_DEBUG"))
        fprintf(stderr, "ds4: [dbg] begin_cmd rot=%u gen=%llu\n",
                slot, (unsigned long long)c.recording_generation);
    return 1;
}

#define DS4_VK_TRACE_KERNEL(name) \
    do { if (getenv("DS4_VULKAN_TRACE_KERNELS")) \
             fprintf(stderr, "ds4: [trace] %s\n", name); } while (0)

static int end_and_submit(void) {
    auto &c = get_cmd_ctx();
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    if (!c.recording) return 1;
    if (c.command_count == 0) {
        VK_CHECK_BOOL(vkEndCommandBuffer(c.cmd));
        c.recording = false;
        return 1;
    }
    if (getenv("DS4_VULKAN_DEBUG"))
        fprintf(stderr, "ds4: [dbg] end_and_submit cc=%u rot=%u gen=%llu\n",
                (unsigned)c.command_count, c.cmd_rot_idx,
                (unsigned long long)c.recording_generation);
    const uint64_t flush_start = timeline_now_ns();
    for (auto &[base, header] : g_vk.tensor_headers) {
        (void)base;
        if (header->host_visible && !header->host_coherent)
            (void)vmaFlushAllocation(g_vk.allocator, header->allocation, 0, header->bytes);
    }
    timeline_duration_current(TimelineEventKind::HostFlush, "flush_live_tensors",
                              flush_start);
    VK_CHECK_BOOL(vkEndCommandBuffer(c.cmd));
    c.recording = false;
    const uint64_t wait_value = c.last_submit_value;
    const uint64_t signal_value = wait_value + 1;
    VkTimelineSemaphoreSubmitInfo timeline{};
    timeline.sType = VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO;
    timeline.waitSemaphoreValueCount = wait_value != 0 ? 1u : 0u;
    timeline.pWaitSemaphoreValues = wait_value != 0 ? &wait_value : nullptr;
    timeline.signalSemaphoreValueCount = 1;
    timeline.pSignalSemaphoreValues = &signal_value;
    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.pNext = &timeline;
    si.waitSemaphoreCount = wait_value != 0 ? 1u : 0u;
    si.pWaitSemaphores = wait_value != 0 ? &c.semaphore : nullptr;
    si.pWaitDstStageMask = wait_value != 0 ? &wait_stage : nullptr;
    si.commandBufferCount = 1; si.pCommandBuffers = &c.cmd;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &c.semaphore;
    const uint64_t submit_start = timeline_now_ns();
    {
        std::lock_guard<std::mutex> lock(g_vk.queue_mutex);
        VK_CHECK_BOOL(vkQueueSubmit(g_vk.queue, 1, &si, VK_NULL_HANDLE));
    }
    if (c.timeline_collecting) {
        TimelineEvent *event = timeline_add(c, TimelineEventKind::Submit, "queue_submit");
        if (event) {
            event->duration_ns = timeline_now_ns() - submit_start;
            event->count = c.command_count;
        }
    }
    const uint32_t slot = c.cmd_rot_idx;
    c.last_submit_value = signal_value;
    c.slot_submit_values[slot] = signal_value;
    c.slot_generations[slot] = c.recording_generation;
    c.slot_timestamp_counts[slot] = c.timestamp_cursor;
    c.slot_routed_timestamps[slot].swap(c.routed_timestamps);
    c.cmd_rot_idx = (slot + 1u) % DS4_VK_COMMAND_RING_SIZE;
    return 1;  /* DS4: non-zero = success */
}

static void invalidate_live_tensors(void) {
    const uint64_t invalidate_start = timeline_now_ns();
    for (auto &[base, header] : g_vk.tensor_headers) {
        (void)base;
        if (header->host_visible && !header->host_coherent)
            (void)vmaInvalidateAllocation(g_vk.allocator, header->allocation, 0, header->bytes);
    }
    timeline_duration_current(TimelineEventKind::HostInvalidate,
                              "invalidate_live_tensors",
                              invalidate_start);
}

static int wait_cmd(void) {
    auto &c = get_cmd_ctx();
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    if (!wait_for_submit_value(c, c.last_submit_value, "wait_cmd")) return 0;
    invalidate_live_tensors();
    return 1;  /* DS4: non-zero = success */
}

static int submit_and_wait_force(void) {
    int r = end_and_submit(); if (!r) return 0;
    return wait_cmd();
}

static int submit_and_wait(void) {
    if (get_cmd_ctx().layer_batch_active) return 1;
    return submit_and_wait_force();
}

static int defer_layer_batch_resources(VulkanCommandCtx &ctx) {
    /* A long prefill layer may cross command-ring submissions through
     * maybe_submit(). Associate the entire scope with the final submission:
     * every later submission waits on the preceding timeline value, so the
     * final slot cannot complete before any earlier slot that referenced
     * these resources. Retaining them slightly longer is intentional and
     * avoids a per-submission ownership list. */
    uint32_t slot = DS4_VK_COMMAND_RING_SIZE;
    for (uint32_t candidate = 0;
         candidate < DS4_VK_COMMAND_RING_SIZE; candidate++) {
        if (ctx.slot_submit_values[candidate] == ctx.last_submit_value) {
            slot = candidate;
            break;
        }
    }
    if (slot == DS4_VK_COMMAND_RING_SIZE) return 0;
    auto &descriptors = ctx.slot_descriptors[slot];
    descriptors.insert(descriptors.end(), ctx.layer_batch_descriptors.begin(),
                       ctx.layer_batch_descriptors.end());
    auto &tensors = ctx.slot_tensors[slot];
    tensors.insert(tensors.end(), ctx.layer_batch_tensors.begin(),
                   ctx.layer_batch_tensors.end());
    auto &in_place_ptrs = ctx.slot_in_place_ptrs[slot];
    in_place_ptrs.insert(in_place_ptrs.end(), ctx.layer_batch_in_place_ptrs.begin(),
                         ctx.layer_batch_in_place_ptrs.end());
    ctx.layer_batch_descriptors.clear();
    ctx.layer_batch_tensors.clear();
    ctx.layer_batch_in_place_ptrs.clear();
    return 1;
}

static int retire_layer_batch_span(VulkanCommandCtx &ctx, bool resume) {
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    const bool was_active = ctx.layer_batch_active;
    ctx.layer_batch_active = false;
    /* Selected-layer capture normally fences at layer end so timestamp
     * queries and resources are immediately retired.  The production decode
     * comparison needs the command-ring behavior instead; this explicit
     * diagnostic gate preserves that behavior and summarizes after the
     * enclosing token completion. */
    const bool nonblocking_timeline = ctx.layer_timeline_active &&
        getenv("DS4_VULKAN_TIMELINE_LAYER_NO_WAIT") != nullptr;
    const bool defer = ctx.command_count != 0 && !resume &&
        (!ctx.layer_timeline_active || nonblocking_timeline) &&
        command_ring_enabled();
    int ok = defer ? end_and_submit() : submit_and_wait_force();
    if (ok && defer) ok = defer_layer_batch_resources(ctx);
    if (defer) return ok;
    const uint64_t descriptor_start = timeline_now_ns();
    for (VkDescriptorSet set : ctx.layer_batch_descriptors) {
        if (vkFreeDescriptorSets(g_vk.device, g_vk.desc_pool, 1, &set) != VK_SUCCESS)
            ok = 0;
    }
    if (!ctx.layer_batch_descriptors.empty())
        timeline_duration_current(TimelineEventKind::DescriptorFree,
                                  "descriptor_free_cpu", descriptor_start);
    for (ds4_gpu_tensor *tensor : ctx.layer_batch_tensors)
        ds4_gpu_tensor_free(tensor);
    for (void *ptr : ctx.layer_batch_in_place_ptrs) {
        auto it = g_vk.tensor_headers.find(ptr);
        if (it == g_vk.tensor_headers.end()) continue;
        if (!release_tensor_header(it->second))
            g_vk.tensor_headers.erase(it);
    }
    ctx.layer_batch_descriptors.clear();
    ctx.layer_batch_tensors.clear();
    ctx.layer_batch_in_place_ptrs.clear();
    if (ok && resume && !ctx.recording) ok = begin_cmd();
    ctx.layer_batch_active = was_active && resume && ok;
    return ok;
}

static uint32_t command_submit_limit(void) {
    const char *env = getenv("DS4_VULKAN_SUBMIT_COMMANDS");
    if (!env || !env[0]) return 64;
    const unsigned long parsed = strtoul(env, nullptr, 10);
    return parsed >= 8 && parsed <= 1024 ? (uint32_t)parsed : 64;
}

/* Split long command buffers into multiple submissions (llama.cpp-style):
 * RADV can crash finalizing a huge CS right after a large prefill, and
 * in-flight weight eviction is bounded by keeping command buffers short. */
static void maybe_submit(void) {
    auto &c = get_cmd_ctx();
    if (c.recording && c.command_count >= command_submit_limit()) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr, "ds4: [dbg] maybe_submit cc=%u\n", (unsigned)c.command_count);
        end_and_submit();
        begin_cmd();
    }
}

/* ---- Compute Dispatch ---- */

static void mark_bound_weight_buffers(VkDescriptorBufferInfo *buffers,
                                      uint32_t count, uint64_t generation);

static int dispatch_shader(const char *name,
                           const void *push, uint32_t push_size,
                           VkDescriptorBufferInfo *bufs, uint32_t n_bufs,
                           uint32_t gx, uint32_t gy, uint32_t gz)
{
    auto it = g_vk.shader_map.find(name);
    if (it == g_vk.shader_map.end()) return -1;
    auto &e = g_vk.shaders[it->second];
    auto &c = get_cmd_ctx();

    vkCmdBindPipeline(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, e.pipeline);

    /* Allocate + update descriptor set */
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = g_vk.desc_pool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &e.desc_layout;
    VkDescriptorSet ds;
    VK_CHECK_RAW(vkAllocateDescriptorSets(g_vk.device, &dai, &ds));

    std::vector<VkWriteDescriptorSet> writes(n_bufs);
    for (uint32_t i = 0; i < n_bufs; i++) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds; writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bufs[i];
    }
    if (n_bufs) vkUpdateDescriptorSets(g_vk.device, n_bufs, writes.data(), 0, nullptr);
    mark_bound_weight_buffers(bufs, n_bufs, c.recording_generation);
    timeline_descriptors(c, name, bufs, n_bufs);

    vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            e.layout, 0, 1, &ds, 0, nullptr);

    if (push && push_size)
        vkCmdPushConstants(c.cmd, e.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);

    timeline_dispatch(c, name, bufs, n_bufs, gx, gy, gz);

    /* Reset descriptor pool periodically (simplified: reset each time) */
    /* In production, use multiple pools or recycle sets */
    maybe_submit();
    return 0;
}

/* =====================================================================
 * PART 2: GPU API Implementations
 * ===================================================================== */

/* ---- Initialization ---- */
int ds4_gpu_init(void) {
    if (g_vk.initialized) return 1;

    VkApplicationInfo app{};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "DS4"; app.applicationVersion = VK_MAKE_API_VERSION(0,1,0,0);
    app.pEngineName = "DS4 Vulkan"; app.engineVersion = VK_MAKE_API_VERSION(0,1,0,0);
    app.apiVersion = VK_API_VERSION_1_3;
    const char *ext[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME,
    };
    VkInstanceCreateInfo ici{};
    ici.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = 2; ici.ppEnabledExtensionNames = ext;

    VkResult res = vkCreateInstance(&ici, nullptr, &g_vk.instance);
    if (res != VK_SUCCESS) { fprintf(stderr, "ds4: VULKAN instance failed (%d)\n", res); return 0; }
    g_vk.pfnGetMemoryHostPointerProperties =
        (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetInstanceProcAddr(
            g_vk.instance, "vkGetMemoryHostPointerPropertiesEXT");
    if (select_physical_device() != 0) return 0;
    if (create_logical_device() != 0) return 0;
    load_all_shaders();
    const char *bg = getenv("DS4_VULKAN_WEIGHT_BUDGET_GB");
    if (bg && *bg) g_vk.weight_budget = (uint64_t)atoll(bg) * 1024ull * 1024ull * 1024ull;
    g_vk.initialized = true;
    fprintf(stderr, "ds4: VULKAN backend ready\n");
    return 1;  /* DS4 convention: 1 = success, 0 = failure */
}

void ds4_gpu_cleanup(void) {
    if (!g_vk.initialized) return;
    timeline_device_wait_idle("cleanup");
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    for (auto &[_, c] : g_vk.cmd_ctxs) {
        (void)retire_completed_slots(c, c.last_submit_value);
        timeline_dump(c);
        for (VkQueryPool pool : c.timestamp_pools)
            if (pool) vkDestroyQueryPool(g_vk.device, pool, nullptr);
        if (c.semaphore) vkDestroySemaphore(g_vk.device, c.semaphore, nullptr);
        if (c.pool) vkDestroyCommandPool(g_vk.device, c.pool, nullptr);
    }
    g_vk.cmd_ctxs.clear();
    for (auto &e : g_vk.shaders) {
        if (e.pipeline) vkDestroyPipeline(g_vk.device, e.pipeline, nullptr);
        if (e.layout) vkDestroyPipelineLayout(g_vk.device, e.layout, nullptr);
        if (e.desc_layout) vkDestroyDescriptorSetLayout(g_vk.device, e.desc_layout, nullptr);
        if (e.module) vkDestroyShaderModule(g_vk.device, e.module, nullptr);
    }
    g_vk.shaders.clear(); g_vk.shader_map.clear();
    /* Free all tracked tensor allocations */
    for (auto &[_, h] : g_vk.tensor_headers) {
        if (h->buffer) vmaDestroyBuffer(g_vk.allocator, h->buffer, h->allocation);
        if (h->device_local_scratch) free(h->mapped_ptr);
        free(h);
    }
    g_vk.tensor_headers.clear();
    g_vk.scratch_pool.clear();
    g_vk.device_scratch_pool.clear();
    for (auto &[_, e] : g_vk.weight_cache)
        if (e.buffer) vmaDestroyBuffer(g_vk.allocator, e.buffer, e.allocation);
    g_vk.weight_cache.clear();
    for (auto &[_, e] : g_vk.aligned_cache)
        if (e.gpu.buffer) vmaDestroyBuffer(g_vk.allocator, e.gpu.buffer, e.gpu.allocation);
    g_vk.aligned_cache.clear();
    g_vk.range_registry.clear();
    /* Destroy the external-host-memory model buffer (no heap budget) */
    if (g_vk.model_buffer) vkDestroyBuffer(g_vk.device, g_vk.model_buffer, nullptr);
    if (g_vk.model_mem) vkFreeMemory(g_vk.device, g_vk.model_mem, nullptr);
    g_vk.model_buffer = VK_NULL_HANDLE;
    g_vk.model_mem = VK_NULL_HANDLE;
    /* Destroy descriptor pool */
    if (g_vk.desc_pool) vkDestroyDescriptorPool(g_vk.device, g_vk.desc_pool, nullptr);
    /* Destroy VMA allocator - skip if any leaks remain (the VkDevice teardown
     * will free all GPU memory regardless).  VMA debug builds assert on leaks. */
    if (g_vk.allocator) {
        VmaTotalStatistics vma_stats;
        vmaCalculateStatistics(g_vk.allocator, &vma_stats);
        if (vma_stats.total.statistics.allocationCount == 0 ||
            getenv("DS4_VULKAN_FORCE_CLEANUP") != NULL) {
            VmaAllocator a = g_vk.allocator;
            g_vk.allocator = VK_NULL_HANDLE;
            vmaDestroyAllocator(a);
        } else {
            fprintf(stderr, "ds4: VULKAN skipping VMA destroy (%u allocations still live)\n",
                    (unsigned)vma_stats.total.statistics.allocationCount);
            g_vk.allocator = VK_NULL_HANDLE;
        }
    }
    if (g_vk.device) vkDestroyDevice(g_vk.device, nullptr);
    if (g_vk.instance) vkDestroyInstance(g_vk.instance, nullptr);
    memset(&g_vk, 0, sizeof(g_vk));
}

void ds4_vulkan_get_caps(ds4_vulkan_caps *caps) { if (caps) *caps = g_vk.caps; }

/* ---- Tensor Management ---- */

static ds4_gpu_tensor *alloc_tensor_kind(uint64_t bytes, bool device_local,
                                         bool reusable_scratch) {
    if (!bytes) return nullptr;
    ds4_gpu_tensor *t = (ds4_gpu_tensor*)calloc(1, sizeof(ds4_gpu_tensor));
    if (!t) return nullptr;

    auto &pool = device_local ? g_vk.device_scratch_pool : g_vk.scratch_pool;
    if (reusable_scratch) {
        auto pit = pool.find(bytes);
        if (pit != pool.end() && !pit->second.empty()) {
            TensorHeader *h = pit->second.back();
            pit->second.pop_back();
            t->ptr = h->mapped_ptr;
            t->bytes = bytes;
            t->owner = 1;
            h->bytes = bytes;
            if (getenv("DS4_VULKAN_TIMELINE"))
                timeline_resource(get_cmd_ctx(), TimelineEventKind::TensorAlloc,
                                  device_local ? "device_scratch_reuse" : "scratch_reuse",
                                  bytes);
            return t;
        }
    }

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = device_local ? VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
                             : VMA_MEMORY_USAGE_AUTO;
    aci.flags = device_local ? 0u
                             : (VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT |
                                VMA_ALLOCATION_CREATE_MAPPED_BIT);
    VkBuffer buf; VmaAllocation alloc;
    VmaAllocationInfo ai;
    VkResult res = vmaCreateBuffer(g_vk.allocator, &bci, &aci, &buf, &alloc, &ai);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "ds4: VULKAN tensor alloc failed for %lu bytes: VkResult=%d\n",
                (unsigned long)bytes, (int)res);
        free(t); return nullptr;
    }
    /* Device-local scratch is shader-only.  Keep a stable identity token for
     * the legacy pointer-keyed tensor registry; no CPU path may dereference
     * it, and flush/invalidate explicitly skip non-host-visible allocations. */
    t->ptr = device_local ? malloc(1) : ai.pMappedData;
    if (!t->ptr) {
        vmaDestroyBuffer(g_vk.allocator, buf, alloc);
        free(t);
        return nullptr;
    }
    t->bytes = bytes; t->owner = 1;

    TensorHeader *h = (TensorHeader*)calloc(1, sizeof(TensorHeader));
    if (!h) {
        vmaDestroyBuffer(g_vk.allocator, buf, alloc);
        if (device_local) free(t->ptr);
        free(t);
        return nullptr;
    }
    h->buffer = buf; h->allocation = alloc; h->bytes = bytes;
    h->mapped_ptr = t->ptr;
    h->reusable_scratch = reusable_scratch;
    h->device_local_scratch = device_local;
    h->host_visible = !device_local;
    VkMemoryPropertyFlags memory_properties = 0;
    vmaGetAllocationMemoryProperties(g_vk.allocator, alloc, &memory_properties);
    h->host_coherent = (memory_properties & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    g_vk.tensor_headers[t->ptr] = h;
    if (getenv("DS4_VULKAN_TIMELINE"))
        timeline_resource(get_cmd_ctx(), TimelineEventKind::TensorAlloc,
                          "tensor", bytes);
    return t;
}

static bool release_tensor_header(TensorHeader *header) {
    if (!header) return false;
    if (header->reusable_scratch) {
        auto &pool = header->device_local_scratch
            ? g_vk.device_scratch_pool : g_vk.scratch_pool;
        pool[header->bytes].push_back(header);
        return true;
    }
    if (header->buffer)
        vmaDestroyBuffer(g_vk.allocator, header->buffer, header->allocation);
    free(header);
    return false;
}

static ds4_gpu_tensor *ds4_gpu_tensor_alloc_device_scratch(uint64_t bytes) {
    return alloc_tensor_kind(bytes, true, true);
}

static int ds4_gpu_tensor_alloc_host_scratch_in_place(ds4_gpu_tensor *t,
                                                       uint64_t bytes) {
    if (!t) return 1;
    ds4_gpu_tensor *a = alloc_tensor_kind(bytes, false, true);
    if (!a) return 2;
    *t = *a;
    free(a);
    return 0;
}

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    return alloc_tensor_kind(bytes, false, false);
}

static int ensure_weight(uint64_t offset, uint64_t needed_bytes);
static bool find_tensor_buffer(const ds4_gpu_tensor *tensor,
                               VkBuffer &buffer, VkDeviceSize &offset);
static bool find_model_buffer(uint64_t offset, uint64_t bytes,
                              VkBuffer &buffer, VkDeviceSize &buffer_offset,
                              VkDeviceSize &range);

ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed(uint64_t bytes) {
    return ds4_gpu_tensor_alloc(bytes);
}

ds4_gpu_tensor *ds4_gpu_tensor_view(const ds4_gpu_tensor *base, uint64_t offset, uint64_t bytes) {
    if (!base || !base->ptr || offset + bytes > base->bytes) return nullptr;
    ds4_gpu_tensor *t = (ds4_gpu_tensor*)calloc(1, sizeof(ds4_gpu_tensor));
    if (!t) return nullptr;
    t->ptr = (char*)base->ptr + offset; t->bytes = bytes; t->owner = 0;
    return t;
}

void ds4_gpu_tensor_free(ds4_gpu_tensor *t) {
    if (!t) return;
    auto &ctx = get_cmd_ctx();
    if (ctx.layer_batch_active && t->owner && t->ptr) {
        ctx.layer_batch_tensors.push_back(t);
        return;
    }
    if (t->owner && t->ptr) {
        auto it = g_vk.tensor_headers.find(t->ptr);
        if (it != g_vk.tensor_headers.end()) {
            if (getenv("DS4_VULKAN_TIMELINE"))
                timeline_resource(get_cmd_ctx(), TimelineEventKind::TensorFree,
                                  "tensor", it->second->bytes);
            if (!release_tensor_header(it->second))
                g_vk.tensor_headers.erase(it);
        }
    }
    free(t);
}

uint64_t ds4_gpu_tensor_bytes(const ds4_gpu_tensor *t) { return t ? t->bytes : 0; }
void *ds4_gpu_tensor_contents(ds4_gpu_tensor *t) { return t ? t->ptr : nullptr; }

int ds4_gpu_tensor_fill_f32(ds4_gpu_tensor *t, float value, uint64_t count) {
    if (!t || !t->ptr || !g_vk.initialized) return 0;
    float *d = (float*)t->ptr;
    for (uint64_t i = 0; i < count && i < t->bytes/4; i++) d[i] = value;
    return 1;
}

int ds4_gpu_tensor_write(ds4_gpu_tensor *t, uint64_t off, const void *data, uint64_t bytes) {
    if (!t || !t->ptr || off + bytes > t->bytes) return 0;
    memcpy((char*)t->ptr + off, data, bytes);
    return 1;
}

int ds4_gpu_tensor_read(const ds4_gpu_tensor *t, uint64_t off, void *data, uint64_t bytes) {
    if (!t || !t->ptr || off + bytes > t->bytes) return 0;
    memcpy(data, (const char*)t->ptr + off, bytes);
    return 1;
}

int ds4_gpu_tensor_copy(ds4_gpu_tensor *dst, uint64_t doff,
                         const ds4_gpu_tensor *src, uint64_t soff, uint64_t bytes) {
    if (!dst || !dst->ptr || !src || !src->ptr) return 0;
    if (doff + bytes > dst->bytes || soff + bytes > src->bytes) return 0;
    memcpy((char*)dst->ptr + doff, (const char*)src->ptr + soff, bytes);
    return 1;
}

int ds4_gpu_tensor_copy_f32_to_f16(ds4_gpu_tensor *dst, uint64_t doff,
                                    const ds4_gpu_tensor *src, uint64_t soff, uint64_t count) {
    if (!dst || !dst->ptr || !src || !src->ptr) return 0;
    const float *s = (const float*)((const char*)src->ptr + soff);
    uint16_t *d = (uint16_t*)((char*)dst->ptr + doff);
    for (uint64_t i = 0; i < count; i++) {
        uint32_t bits; memcpy(&bits, &s[i], 4);
        uint16_t sign = (bits >> 16) & 0x8000u;
        int exp = ((bits >> 23) & 0xff) - 127 + 15;
        uint32_t mant = bits & 0x7fffffu;
        if (exp <= 0) d[i] = sign;
        else if (exp >= 31) d[i] = (uint16_t)(sign | 0x7c00 | (mant >> 13));
        else d[i] = (uint16_t)(sign | ((uint16_t)exp << 10) | (mant >> 13));
    }
    return 1;
}

/* ---- Commands ---- */

int ds4_gpu_begin_commands(void) { return begin_cmd(); }
int ds4_gpu_commands_active(void) {
    auto &ctx = get_cmd_ctx();
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    return ctx.recording ? 1 : 0;
}
int ds4_gpu_flush_commands(void) {
    /* The engine keeps recording after a flush (e.g. SSD streaming async
     * loads), so start a fresh command buffer like Metal's next encoder. */
    auto &ctx = get_cmd_ctx();
    if (ctx.layer_batch_active)
        return retire_layer_batch_span(ctx, true);
    int ok = end_and_submit();
    if (ok) ok = begin_cmd();
    return ok;
}
int ds4_gpu_end_commands(void) {
    int ok = submit_and_wait();
    auto &ctx = get_cmd_ctx();
    if (ok && ctx.layer_timeline_end_pending)
        timeline_layer_summary(ctx);
    if (ok && ctx.timeline_captured_dispatches >= ctx.timeline_max_dispatches)
        timeline_dump(ctx);
    return ok;
}
int ds4_gpu_synchronize(void) {
    VK_CHECK_BOOL(timeline_device_wait_idle("synchronize"));
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    for (auto &[_, ctx] : g_vk.cmd_ctxs)
        if (!retire_completed_slots(ctx, ctx.last_submit_value)) return 0;
    invalidate_live_tensors();
    return 1;
}

extern "C" void ds4_gpu_timeline_layer_begin(uint32_t layer) {
    const char *target = getenv("DS4_VULKAN_TIMELINE_LAYER");
    if (!target || !target[0] || strtoul(target, nullptr, 10) != layer) return;
    auto &ctx = get_cmd_ctx();
    if (ctx.layer_timeline_active) return;
    ctx.timeline_events.clear();
    ctx.timeline_events.reserve(2048);
    ctx.timeline_enabled = true;
    ctx.timeline_collecting = true;
    ctx.timeline_dumped = false;
    ctx.timeline_seen_dispatches = 0;
    ctx.timeline_skip_dispatches = 0;
    ctx.timeline_max_dispatches = DS4_VK_TIMELINE_MAX_DISPATCHES;
    ctx.timeline_captured_dispatches = 0;
    ctx.layer_timeline_active = true;
    ctx.layer_timeline_end_pending = false;
    ctx.layer_timeline_layer = layer;
    ctx.layer_timeline_start_ns = timeline_now_ns();
    ctx.layer_timeline_stop_ns = 0;
    ctx.layer_timeline_stage_cursor = 0;
}

extern "C" void ds4_gpu_timeline_layer_end(uint32_t layer) {
    const char *target = getenv("DS4_VULKAN_TIMELINE_LAYER");
    if (!target || !target[0] || strtoul(target, nullptr, 10) != layer) return;
    auto &ctx = get_cmd_ctx();
    if (!ctx.layer_timeline_active || ctx.layer_timeline_layer != layer) return;
    if (ctx.recording || getenv("DS4_VULKAN_TIMELINE_LAYER_NO_WAIT")) {
        /* Prefill has no decode layer batch and can still be recording here.
         * Nonblocking decode capture also reaches this branch: stop admitting
         * later dispatch events, then summarize after token completion in
         * ds4_gpu_end_commands(). */
        ctx.layer_timeline_end_pending = true;
        ctx.layer_timeline_stop_ns = timeline_now_ns();
        ctx.timeline_collecting = false;
        return;
    }
    timeline_layer_summary(ctx);
}

extern "C" void ds4_gpu_timeline_stage_end(const char *stage) {
    const char *target = getenv("DS4_VULKAN_TIMELINE_LAYER");
    if (!target || !target[0]) return;
    auto &ctx = get_cmd_ctx();
    if (!ctx.layer_timeline_active) return;
    for (size_t i = ctx.layer_timeline_stage_cursor;
         i < ctx.timeline_events.size(); i++)
        ctx.timeline_events[i].stage = stage;
    ctx.layer_timeline_stage_cursor = ctx.timeline_events.size();
}

extern "C" int ds4_gpu_batch_layer_begin(uint32_t layer) {
    (void)layer;
    auto &ctx = get_cmd_ctx();
    if (ctx.layer_batch_active) return 0;
    if (ctx.recording && ctx.command_count != 0) {
        /* Decode records token embedding before opening the first layer
         * lifetime scope.  The timeline wait attached to the next submit
         * already orders that work; waiting here needlessly idles the host
         * once the command ring is enabled.  Keep the old fence boundary in
         * serial mode, where it remains the diagnostic lifetime contract. */
        const int ok = command_ring_enabled()
            ? end_and_submit() : submit_and_wait_force();
        if (!ok) return 0;
    }
    if (!ctx.recording && !begin_cmd()) return 0;
    ctx.layer_batch_descriptors.clear();
    ctx.layer_batch_tensors.clear();
    ctx.layer_batch_in_place_ptrs.clear();
    ctx.layer_batch_descriptors.reserve(128);
    ctx.layer_batch_tensors.reserve(32);
    ctx.layer_batch_in_place_ptrs.reserve(16);
    ctx.layer_batch_active = true;
    return 1;
}

extern "C" int ds4_gpu_batch_prefill_layer_begin(uint32_t layer) {
    (void)layer;
    auto &ctx = get_cmd_ctx();
    if (ctx.layer_batch_active) return 0;
    /* Prefill already owns an ordered layer-major stream.  Attaching the
     * lifetime scope must not submit/wait on the preceding layer; the next
     * submission carries the existing timeline wait, and Vulkan's in-command
     * hazards preserve ordering. */
    if (!ctx.recording && !begin_cmd()) return 0;
    ctx.layer_batch_descriptors.clear();
    ctx.layer_batch_tensors.clear();
    ctx.layer_batch_in_place_ptrs.clear();
    ctx.layer_batch_descriptors.reserve(128);
    ctx.layer_batch_tensors.reserve(32);
    ctx.layer_batch_in_place_ptrs.reserve(16);
    ctx.layer_batch_active = true;
    return 1;
}

extern "C" int ds4_gpu_batch_layer_end(uint32_t layer) {
    (void)layer;
    auto &ctx = get_cmd_ctx();
    if (!ctx.layer_batch_active) return 1;
    return retire_layer_batch_span(ctx, false);
}

int ds4_gpu_signal_selected_readback_ready(uint64_t *ev) {
    auto &c = get_cmd_ctx(); *ev = ++c.event_counter; return 1;
}

int ds4_gpu_commit_and_wait_selected_readback(uint64_t ev, const char *label) {
    (void)ev;
    /* End + wait + re-begin: the engine reads the selected ids on the CPU and
     * then keeps encoding GPU kernels (routed MoE) without a begin_commands. */
    auto &ctx = get_cmd_ctx();
    const uint64_t readback_start = timeline_now_ns();
    if (ctx.layer_batch_active)
        return retire_layer_batch_span(ctx, true);
    int ok = end_and_submit();
    if (ok) ok = wait_cmd();
    if (ok) ok = begin_cmd();
    if (getenv("DS4_VULKAN_TIMELINE"))
        timeline_duration_current(TimelineEventKind::Wait,
                                  label && label[0] ? label : "selected_readback",
                                  readback_start);
    return ok;
}

int ds4_gpu_wait_selected_readback_ready(uint64_t ev, const char *label) {
    (void)ev;
    const uint64_t start = timeline_now_ns();
    const int ok = wait_cmd();
    if (getenv("DS4_VULKAN_TIMELINE"))
        timeline_duration_current(TimelineEventKind::Wait,
                                  label && label[0] ? label : "selected_readback_wait",
                                  start);
    return ok;
}

/* ---- Model Loading ---- */

static void clear_weight_cache(void) {
    if (!g_vk.weight_cache.empty()) {
        (void)timeline_device_wait_idle("weight_cache_clear");
        for (auto &[offset, entry] : g_vk.weight_cache) {
            timeline_resource_current(TimelineEventKind::BufferFree,
                                      "weight_cache_clear", entry.size, offset);
            vmaDestroyBuffer(g_vk.allocator, entry.buffer, entry.allocation);
        }
        g_vk.weight_cache.clear();
    }
    if (!g_vk.aligned_cache.empty()) {
        (void)timeline_device_wait_idle("aligned_cache_clear");
        for (auto &[offset, entry] : g_vk.aligned_cache) {
            timeline_resource_current(TimelineEventKind::BufferFree,
                                      "aligned_cache_clear", entry.gpu.size, offset);
            vmaDestroyBuffer(g_vk.allocator, entry.gpu.buffer, entry.gpu.allocation);
        }
        g_vk.aligned_cache.clear();
    }
    g_vk.weight_used = 0;
    g_vk.range_registry.clear();
}

static void set_model_map_identity(const void *model_map, uint64_t model_size) {
    if (g_vk.model_map != model_map || g_vk.model_size != model_size)
        clear_weight_cache();
    g_vk.model_map = model_map;
    g_vk.model_size = model_size;
}

int ds4_gpu_set_model_map(const void *m, uint64_t s) {
    if (!m) return 0;
    if (g_vk.model_map == m && g_vk.model_size == s) clear_weight_cache();
    set_model_map_identity(m, s);
    return 1;
}
int ds4_gpu_set_model_fd(int fd) { (void)fd; return 0; }
int ds4_gpu_set_model_fd_for_map(int fd, const void *m) {
    (void)fd;
    if (!m) return 0;
    set_model_map_identity(m, g_vk.model_size);
    return 1;
}

int ds4_gpu_set_model_map_range(const void *m, uint64_t s, uint64_t mo, uint64_t ms, uint64_t mt) {
    (void)mt;
    if (!m || s == 0 || mo > s || ms > s - mo) return 0;
    set_model_map_identity(m, s);
    return 1;  /* DS4 convention: 1 = success */
}

int ds4_gpu_set_model_map_spans(const void *m, uint64_t s, const uint64_t *o, const uint64_t *sz, uint32_t c, uint64_t mt) {
    (void)mt;
    if (!m || s == 0 || !o || !sz || c == 0) return 0;
    for (uint32_t i = 0; i < c; i++) {
        if (o[i] > s || sz[i] == 0 || sz[i] > s - o[i]) return 0;
    }
    set_model_map_identity(m, s);
    return 1;  /* DS4 convention: 1 = success */
}

static int ensure_weight(uint64_t offset, uint64_t needed_bytes);

static bool command_generation_active(uint64_t generation) {
    if (generation == 0) return false;
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    for (const auto &[_, context] : g_vk.cmd_ctxs) {
        if (context.recording &&
            context.recording_generation == generation)
            return true;
        for (uint32_t slot = 0; slot < DS4_VK_COMMAND_RING_SIZE; slot++)
            if (context.slot_submit_values[slot] != 0 &&
                context.slot_generations[slot] == generation)
                return true;
    }
    return false;
}

static void prune_weight_generations(
        decltype(g_vk.weight_cache)::mapped_type &entry) {
    auto &generations = entry.active_generations;
    generations.erase(std::remove_if(generations.begin(), generations.end(),
        [](uint64_t generation) { return !command_generation_active(generation); }),
        generations.end());
}

static void mark_weight_generation(
    decltype(g_vk.weight_cache)::mapped_type &entry,
    uint64_t generation) {
    prune_weight_generations(entry);
    if (command_generation_active(generation) &&
        std::find(entry.active_generations.begin(), entry.active_generations.end(),
                  generation) == entry.active_generations.end())
        entry.active_generations.push_back(generation);
}

    static void mark_weight_generation(
        decltype(g_vk.weight_cache)::mapped_type &entry) {
        auto &ctx = get_cmd_ctx();
        if (ctx.recording)
        mark_weight_generation(entry, ctx.recording_generation);
    }

static bool weight_entry_in_use(
        decltype(g_vk.weight_cache)::mapped_type &entry) {
    if (entry.pinned) return true;
    prune_weight_generations(entry);
    return !entry.active_generations.empty();
}

static void mark_bound_weight_buffers(VkDescriptorBufferInfo *buffers,
                                      uint32_t count, uint64_t generation) {
    for (uint32_t i = 0; i < count; i++) {
        for (auto &[_, entry] : g_vk.weight_cache)
            if (entry.buffer == buffers[i].buffer)
                mark_weight_generation(entry, generation);
        for (auto &[_, entry] : g_vk.aligned_cache)
            if (entry.gpu.buffer == buffers[i].buffer)
                mark_weight_generation(entry.gpu, generation);
    }
}

int ds4_gpu_cache_model_range(const void *m, uint64_t s, uint64_t off, uint64_t bytes, const char *label) {
    (void)label;
    if (!m || s == 0 || bytes == 0 || off > s || bytes > s - off) return 0;
    set_model_map_identity(m, s);
    g_vk.range_registry[off] = bytes;
    for (auto &[base, entry] : g_vk.weight_cache) {
        if (off >= base && off - base <= entry.size && bytes <= entry.size - (off - base)) {
            entry.pinned = true;
            break;
        }
    }
    return 1;
}

/* Ensure the weight range covering `offset` (at least `needed` bytes) is
 * resident in a GPU buffer, uploading it from the model mmap on first use.
 * Evicts least-recently-used ranges when g_vk.weight_budget is exceeded. */
static int ensure_weight(uint64_t offset, uint64_t needed_bytes) {
    for (auto &[base, e] : g_vk.weight_cache) {
        if (offset >= base && offset - base <= e.size &&
            needed_bytes <= e.size - (offset - base)) {
            e.last_used = ++g_vk.lru_counter;
            mark_weight_generation(e);
            return 1;
        }
    }
    if (!g_vk.model_map || needed_bytes == 0) return 0;

    uint64_t size = needed_bytes;
    /* Never read past the end of the file-backed mmap (SIGBUS otherwise). */
    if (offset >= g_vk.model_size) return 0;
    if (size > g_vk.model_size - offset) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr, "ds4: [dbg] ensure_weight clamp %llu -> %llu bytes @ %llu\n",
                    (unsigned long long)size,
                    (unsigned long long)(g_vk.model_size - offset),
                    (unsigned long long)offset);
        size = g_vk.model_size - offset;
    }
    if (size == 0) return 0;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VkBuffer buf; VmaAllocation alloc; VmaAllocationInfo ai;
    if (vmaCreateBuffer(g_vk.allocator, &bci, &aci, &buf, &alloc, &ai) != VK_SUCCESS) {
        fprintf(stderr, "ds4: VULKAN ensure_weight: alloc failed (%llu bytes @ %llu); try DS4_VULKAN_WEIGHT_BUDGET_GB\n",
                (unsigned long long)size, (unsigned long long)offset);
        return 0;
    }
    timeline_resource_current(TimelineEventKind::BufferAlloc,
                              "weight_cache", size, offset);

    static VkCommandPool load_pool = VK_NULL_HANDLE;
    if (load_pool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo cpci{};
        cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = g_vk.queue_family;
        cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        vkCreateCommandPool(g_vk.device, &cpci, nullptr, &load_pool);
    }

    /* Staging buffer: the destination is device-local even on iGPUs, so the
     * copy goes host -> staging -> vkCmdCopyBuffer -> device. */
    VkBufferCreateInfo sbci{};
    sbci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    sbci.size = size;
    sbci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo saci{};
    saci.usage = VMA_MEMORY_USAGE_AUTO;
    saci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo sai;
    VkBuffer sbuf; VmaAllocation salloc;
    bool staged = false;
    if (vmaCreateBuffer(g_vk.allocator, &sbci, &saci, &sbuf, &salloc, &sai) == VK_SUCCESS
        && sai.pMappedData) {
        timeline_resource_current(TimelineEventKind::BufferAlloc,
                                  "weight_staging", size, offset);
        memcpy(sai.pMappedData, (const char*)g_vk.model_map + offset, (size_t)size);
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = load_pool;
        cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb;
        if (vkAllocateCommandBuffers(g_vk.device, &cbai, &cb) == VK_SUCCESS) {
            VkCommandBufferBeginInfo bi{};
            bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            if (vkBeginCommandBuffer(cb, &bi) == VK_SUCCESS) {
                VkBufferCopy copy{}; copy.size = size;
                vkCmdCopyBuffer(cb, sbuf, buf, 1, &copy);
                if (vkEndCommandBuffer(cb) == VK_SUCCESS) {
                    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
                    VkFence fence;
                    if (vkCreateFence(g_vk.device, &fci, nullptr, &fence) == VK_SUCCESS) {
                        VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                        si.commandBufferCount = 1; si.pCommandBuffers = &cb;
                        const uint64_t submit_start = timeline_now_ns();
                        VkResult submit_result;
                        {
                            std::lock_guard<std::mutex> lock(g_vk.queue_mutex);
                            submit_result = vkQueueSubmit(g_vk.queue, 1, &si, fence);
                        }
                        if (submit_result == VK_SUCCESS) {
                            timeline_duration_current(TimelineEventKind::Submit,
                                                      "weight_upload", submit_start);
                            const uint64_t wait_start = timeline_now_ns();
                            vkWaitForFences(g_vk.device, 1, &fence, VK_TRUE, UINT64_MAX);
                            timeline_duration_current(TimelineEventKind::Wait,
                                                      "weight_upload", wait_start);
                            staged = true;
                        }
                        vkDestroyFence(g_vk.device, fence, nullptr);
                    }
                }
                vkFreeCommandBuffers(g_vk.device, load_pool, 1, &cb);
            }
        }
        timeline_resource_current(TimelineEventKind::BufferFree,
                                  "weight_staging", size, offset);
        vmaDestroyBuffer(g_vk.allocator, sbuf, salloc);
    }
    if (!staged) {
        timeline_resource_current(TimelineEventKind::BufferFree,
                                  "weight_cache_failed", size, offset);
        vmaDestroyBuffer(g_vk.allocator, buf, alloc);
        return 0;
    }

    g_vk.weight_cache[offset] = {
        buf, alloc, size, ++g_vk.lru_counter, false, {}, {}};
    mark_weight_generation(g_vk.weight_cache[offset]);
    g_vk.weight_used += size;

    /* LRU eviction (never evict the range just uploaded, nor any range still
     * referenced by a recording or submitted command buffer). */
    while (g_vk.weight_used > g_vk.weight_budget) {
        uint64_t lru_base = UINT64_MAX, lru_time = UINT64_MAX;
        for (auto &[b, e] : g_vk.weight_cache) {
            if (b == offset) continue;
            if (weight_entry_in_use(e)) continue;
            if (e.last_used < lru_time) { lru_time = e.last_used; lru_base = b; }
        }
        if (lru_base == UINT64_MAX) break;
        auto it = g_vk.weight_cache.find(lru_base);
        timeline_resource_current(TimelineEventKind::BufferFree,
                      "weight_cache_evict", it->second.size, lru_base);
        vmaDestroyBuffer(g_vk.allocator, it->second.buffer, it->second.allocation);
        g_vk.weight_used -= it->second.size;
        g_vk.weight_cache.erase(it);
    }
    return 1;
}

static bool current_commands_reference_weights(void) {
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    for (const auto &[_, context] : g_vk.cmd_ctxs)
        if ((context.recording && context.command_count != 0) ||
            context.last_submit_value > context.completed_value) return true;
    return false;
}

static bool ranges_overlap(uint64_t left_offset, uint64_t left_size,
                           uint64_t right_offset, uint64_t right_size) {
    if (left_offset < right_offset) return right_offset - left_offset < left_size;
    return left_offset - right_offset < right_size;
}

static bool remove_raw_weight_overlap(uint64_t offset, uint64_t bytes) {
    bool found = false;
    for (const auto &[base, value] : g_vk.weight_cache)
        if (ranges_overlap(offset, bytes, base, value.size)) found = true;
    if (!found) return true;
    if (current_commands_reference_weights() ||
        timeline_device_wait_idle("weight_overlap_remove") != VK_SUCCESS) return false;
    for (auto it = g_vk.weight_cache.begin(); it != g_vk.weight_cache.end();) {
        if (!ranges_overlap(offset, bytes, it->first, it->second.size)) {
            ++it;
            continue;
        }
        timeline_resource_current(TimelineEventKind::BufferFree,
                      "weight_overlap_remove", it->second.size, it->first);
        vmaDestroyBuffer(g_vk.allocator, it->second.buffer, it->second.allocation);
        g_vk.weight_used -= std::min(g_vk.weight_used, it->second.size);
        it = g_vk.weight_cache.erase(it);
    }
    return true;
}

static bool reserve_aligned_weight_budget(uint64_t bytes, uint64_t protected_offset) {
    if (bytes > g_vk.weight_budget) return false;
    while (g_vk.weight_used > g_vk.weight_budget - bytes) {
        uint64_t victim = UINT64_MAX;
        uint64_t oldest = UINT64_MAX;
        bool victim_aligned = false;
        for (auto &[candidate, value] : g_vk.aligned_cache) {
            if (candidate == protected_offset ||
            weight_entry_in_use(value.gpu)) continue;
            if (value.gpu.last_used < oldest) {
                victim = candidate;
                oldest = value.gpu.last_used;
                victim_aligned = true;
            }
        }
        for (auto &[candidate, value] : g_vk.weight_cache) {
            if (weight_entry_in_use(value) ||
                value.last_used >= oldest) continue;
            victim = candidate;
            oldest = value.last_used;
            victim_aligned = false;
        }
        if (victim == UINT64_MAX) return false;
        if (victim_aligned) {
            auto it = g_vk.aligned_cache.find(victim);
            timeline_resource_current(TimelineEventKind::BufferFree,
                                      "aligned_cache_evict", it->second.gpu.size, victim);
            vmaDestroyBuffer(g_vk.allocator, it->second.gpu.buffer, it->second.gpu.allocation);
            g_vk.weight_used -= it->second.gpu.size;
            g_vk.aligned_cache.erase(it);
        } else {
            auto it = g_vk.weight_cache.find(victim);
            timeline_resource_current(TimelineEventKind::BufferFree,
                                      "weight_cache_evict", it->second.size, victim);
            vmaDestroyBuffer(g_vk.allocator, it->second.buffer, it->second.allocation);
            g_vk.weight_used -= it->second.size;
            g_vk.weight_cache.erase(it);
        }
    }
    return true;
}

static bool upload_aligned_artifact(const ds4_vulkan_q8_aligned_artifact &artifact,
                                    VkBuffer &buf, VmaAllocation &alloc) {
    VkBufferCreateInfo bci{}; bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = artifact.bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VmaAllocationCreateInfo aci{}; aci.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    VmaAllocationInfo ai;
    if (vmaCreateBuffer(g_vk.allocator, &bci, &aci, &buf, &alloc, &ai) != VK_SUCCESS)
        return false;
    timeline_resource_current(TimelineEventKind::BufferAlloc,
                              "aligned_weight", artifact.bytes);
    static VkCommandPool load_pool = VK_NULL_HANDLE;
    if (load_pool == VK_NULL_HANDLE) {
        VkCommandPoolCreateInfo cpci{}; cpci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.queueFamilyIndex = g_vk.queue_family;
        cpci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        if (vkCreateCommandPool(g_vk.device, &cpci, nullptr, &load_pool) != VK_SUCCESS) {
            vmaDestroyBuffer(g_vk.allocator, buf, alloc); return false;
        }
    }
    VkBufferCreateInfo sbci{}; sbci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    sbci.size = artifact.bytes; sbci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo saci{}; saci.usage = VMA_MEMORY_USAGE_AUTO;
    saci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo sai; VkBuffer sbuf = VK_NULL_HANDLE; VmaAllocation salloc = VK_NULL_HANDLE;
    bool ok = vmaCreateBuffer(g_vk.allocator, &sbci, &saci, &sbuf, &salloc, &sai) == VK_SUCCESS;
    if (ok)
        timeline_resource_current(TimelineEventKind::BufferAlloc,
                                  "aligned_staging", artifact.bytes);
    if (ok && sai.pMappedData) memcpy(sai.pMappedData, artifact.data, (size_t)artifact.bytes);
    if (ok) {
        VkCommandBufferAllocateInfo cbai{}; cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = load_pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
        VkCommandBuffer cb; ok = vkAllocateCommandBuffers(g_vk.device, &cbai, &cb) == VK_SUCCESS;
        if (ok) {
            VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            ok = vkBeginCommandBuffer(cb, &bi) == VK_SUCCESS;
            if (ok) { VkBufferCopy copy{}; copy.size = artifact.bytes; vkCmdCopyBuffer(cb, sbuf, buf, 1, &copy); ok = vkEndCommandBuffer(cb) == VK_SUCCESS; }
            VkFence fence = VK_NULL_HANDLE; VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
            if (ok) ok = vkCreateFence(g_vk.device, &fci, nullptr, &fence) == VK_SUCCESS;
            if (ok) {
                VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
                si.commandBufferCount = 1; si.pCommandBuffers = &cb;
                const uint64_t submit_start = timeline_now_ns();
                {
                    std::lock_guard<std::mutex> lock(g_vk.queue_mutex);
                    ok = vkQueueSubmit(g_vk.queue, 1, &si, fence) == VK_SUCCESS;
                }
                if (ok) timeline_duration_current(TimelineEventKind::Submit,
                                                  "aligned_upload", submit_start);
            }
            if (ok) {
                const uint64_t wait_start = timeline_now_ns();
                ok = vkWaitForFences(g_vk.device, 1, &fence, VK_TRUE, UINT64_MAX) == VK_SUCCESS;
                timeline_duration_current(TimelineEventKind::Wait,
                                          "aligned_upload", wait_start);
            }
            if (fence) vkDestroyFence(g_vk.device, fence, nullptr);
            vkFreeCommandBuffers(g_vk.device, load_pool, 1, &cb);
        }
    }
    if (sbuf) {
        timeline_resource_current(TimelineEventKind::BufferFree,
                                  "aligned_staging", artifact.bytes);
        vmaDestroyBuffer(g_vk.allocator, sbuf, salloc);
    }
    if (!ok) {
        timeline_resource_current(TimelineEventKind::BufferFree,
                                  "aligned_weight_failed", artifact.bytes);
        vmaDestroyBuffer(g_vk.allocator, buf, alloc);
        buf = VK_NULL_HANDLE; alloc = VK_NULL_HANDLE;
    }
    return ok;
}

static bool ensure_aligned_weight(const void *model_map, uint64_t model_size,
                                  uint64_t offset, uint64_t in_dim, uint64_t out_dim,
                                  decltype(g_vk.aligned_cache)::mapped_type *&entry) {
    if (g_vk.model_map != model_map || g_vk.model_size != model_size)
        set_model_map_identity(model_map, model_size);
    auto it = g_vk.aligned_cache.find(offset);
    if (it != g_vk.aligned_cache.end()) {
        if (it->second.model_map != model_map || it->second.model_size != model_size ||
            it->second.in_dim != in_dim || it->second.out_dim < out_dim) {
            if (getenv("DS4_VULKAN_DEBUG"))
                fprintf(stderr,
                        "ds4: [dbg] aligned q8 identity mismatch off=%llu "
                        "cached=%llux%llu requested=%llux%llu\n",
                        (unsigned long long)offset,
                        (unsigned long long)it->second.in_dim,
                        (unsigned long long)it->second.out_dim,
                        (unsigned long long)in_dim,
                        (unsigned long long)out_dim);
            return false;
        }
        it->second.gpu.last_used = ++g_vk.lru_counter;
        mark_weight_generation(it->second.gpu);
        entry = &it->second; return true;
    }
    ds4_vulkan_q8_aligned_artifact artifact{};
    if (!ds4_vulkan_q8_aligned_build(&artifact, model_map, model_size, offset, in_dim, out_dim,
                                     g_vk.caps.min_storage_buffer_offset_alignment)) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr, "ds4: [dbg] aligned q8 build failed off=%llu dims=%llux%llu\n",
                    (unsigned long long)offset, (unsigned long long)in_dim,
                    (unsigned long long)out_dim);
        return false;
    }
    const uint64_t raw_bytes = out_dim * artifact.blocks_per_row * 34u;
    if (!remove_raw_weight_overlap(offset, raw_bytes) ||
        !reserve_aligned_weight_budget(artifact.bytes, offset)) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr,
                    "ds4: [dbg] aligned q8 reserve failed off=%llu artifact=%llu "
                    "used=%llu budget=%llu\n",
                    (unsigned long long)offset,
                    (unsigned long long)artifact.bytes,
                    (unsigned long long)g_vk.weight_used,
                    (unsigned long long)g_vk.weight_budget);
        ds4_vulkan_q8_aligned_free(&artifact);
        return false;
    }
    VkBuffer buf = VK_NULL_HANDLE; VmaAllocation alloc = VK_NULL_HANDLE;
    bool ok = upload_aligned_artifact(artifact, buf, alloc);
    if (!ok && getenv("DS4_VULKAN_DEBUG"))
        fprintf(stderr, "ds4: [dbg] aligned q8 upload failed off=%llu bytes=%llu\n",
                (unsigned long long)offset, (unsigned long long)artifact.bytes);
    if (ok) {
        g_vk.aligned_cache[offset] = {model_map, model_size, offset, in_dim, out_dim,
            artifact.blocks_per_row, artifact.scale_bytes, artifact.payload_offset,
            artifact.payload_bytes,
            {buf, alloc, artifact.bytes, ++g_vk.lru_counter, false,
             {}, {}}};
        mark_weight_generation(g_vk.aligned_cache[offset].gpu);
        g_vk.weight_used += artifact.bytes;
        entry = &g_vk.aligned_cache.find(offset)->second;
    }
    ds4_vulkan_q8_aligned_free(&artifact);
    return ok;
}

int ds4_gpu_cache_q8_f16_range(const void *m, uint64_t s, uint64_t off, uint64_t bytes,
                                 uint64_t idim, uint64_t odim, const char *label) {
    (void)label;
    if (!m || s == 0 || bytes == 0 || off > s || bytes > s - off || idim == 0 || odim == 0)
        return 0;
    const uint64_t blocks = (idim + 31u) / 32u;
    if (idim > 8192u || odim > UINT64_MAX / blocks ||
        odim * blocks > UINT64_MAX / 34u || bytes != odim * blocks * 34u) return 0;
    set_model_map_identity(m, s);
    decltype(g_vk.aligned_cache)::mapped_type *entry = nullptr;
    return ensure_aligned_weight(m, s, off, idim, odim, entry) || ensure_weight(off, bytes);
}

void ds4_gpu_release_q8_f16_cache(void) {
    if (g_vk.aligned_cache.empty()) return;
    (void)timeline_device_wait_idle("aligned_cache_release");
    for (auto &[offset, entry] : g_vk.aligned_cache) {
        timeline_resource_current(TimelineEventKind::BufferFree,
                                  "aligned_cache_release", entry.gpu.size, offset);
        vmaDestroyBuffer(g_vk.allocator, entry.gpu.buffer, entry.gpu.allocation);
        g_vk.weight_used -= std::min(g_vk.weight_used, entry.gpu.size);
    }
    g_vk.aligned_cache.clear();
}
int ds4_gpu_pro_q4_expert_table_auto_available(void) { return 0; }

int ds4_gpu_preload_q4_expert_tables(const void *m, uint64_t s,
                                      uint64_t go, uint64_t uo, uint64_t dwo,
                                      uint64_t geb, uint64_t deb, uint32_t n) {
    (void)m; (void)s; (void)go; (void)uo; (void)dwo; (void)geb; (void)deb; (void)n; return 0; }

int ds4_gpu_should_use_managed_kv_cache(uint64_t kvc, uint64_t ctc) {
    (void)kvc; (void)ctc; return 1; }

void ds4_gpu_set_quality(bool q) { g_vk.quality = q; }
void ds4_gpu_set_ssd_streaming(bool s) { g_vk.ssd_streaming = s; }
void ds4_gpu_set_streaming_expert_cache_budget(uint32_t e) { g_vk.expert_cache_budget = e; }
void ds4_gpu_set_streaming_expert_cache_expert_bytes(uint64_t b) { g_vk.expert_cache_expert_bytes = b; }
uint64_t ds4_gpu_recommended_working_set_size(void) { return 85ull * 1024 * 1024 * 1024; }  /* ~85 GiB */
uint32_t ds4_gpu_stream_expert_cache_configured_count(void) { return g_vk.expert_cache_budget; }
uint32_t ds4_gpu_stream_expert_cache_current_count(void) { return g_vk.streamed_experts; }
void ds4_gpu_stream_expert_cache_reset_route_hotness(void) {}
void ds4_gpu_stream_expert_cache_release_resident(void) { g_vk.streamed_experts = 0; }

uint32_t ds4_gpu_stream_expert_cache_budget_for_expert_size(uint64_t gb, uint64_t db) {
    (void)gb; (void)db; return g_vk.expert_cache_budget; }

int ds4_gpu_stream_expert_cache_seed_selected(const ds4_gpu_stream_expert_table *t,
                                               const int32_t *ids, uint32_t n) {
    (void)t; (void)ids; (void)n; return 0; }

int ds4_gpu_stream_expert_cache_begin_selected_load(const ds4_gpu_stream_expert_table *t,
                                                     const int32_t *ids, uint32_t n) {
    (void)t; (void)ids; (void)n; return 0; }

int ds4_gpu_stream_expert_cache_prepare_selected_batch(const ds4_gpu_stream_expert_table *t,
                                                        const int32_t *ids,
                                                        uint32_t nt, uint32_t ns) {
    (void)t; (void)ids; (void)nt; (void)ns; return 0; }

int ds4_gpu_stream_expert_cache_load_layer(const ds4_gpu_stream_expert_table *t) { (void)t; return 0; }

int ds4_gpu_stream_expert_cache_seed_from_layer_selected(const ds4_gpu_stream_expert_table *t,
                                                          const ds4_gpu_tensor *sel,
                                                          uint32_t nt, uint32_t nst, uint32_t ns) {
    (void)t; (void)sel; (void)nt; (void)nst; (void)ns; return 0; }

int ds4_gpu_stream_expert_cache_release_layer_cache(void) { return 0; }

int ds4_gpu_stream_expert_cache_seed_experts(const ds4_gpu_stream_expert_table *t,
                                              const int32_t *eids, const uint32_t *eprio,
                                              uint32_t ne) {
    (void)t; (void)eids; (void)eprio; (void)ne; return 0; }

void ds4_gpu_print_memory_report(const char *label) {
    fprintf(stderr, "ds4: VULKAN [%s] ", label ? label : "");
    if (g_vk.allocator) {
        VmaTotalStatistics s; vmaCalculateStatistics(g_vk.allocator, &s);
        fprintf(stderr, "mem=%lu KB blocks=%u\n",
                (unsigned long)(s.total.statistics.blockBytes / 1024),
                s.total.statistics.blockCount);
    } else fprintf(stderr, "no allocator\n");
}

/* ---- Simple Vulkan compute operations ---- */

static bool find_tensor_buffer(const ds4_gpu_tensor *tensor,
                               VkBuffer &buffer, VkDeviceSize &offset) {
    if (!tensor || !tensor->ptr) return false;
    auto exact = g_vk.tensor_headers.find(tensor->ptr);
    if (exact != g_vk.tensor_headers.end()) {
        buffer = exact->second->buffer;
        offset = 0;
        return true;
    }
    const char *ptr = (const char *)tensor->ptr;
    for (auto &[base, header] : g_vk.tensor_headers) {
        const char *begin = (const char *)base;
        if (ptr >= begin && ptr < begin + header->bytes) {
            buffer = header->buffer;
            offset = (VkDeviceSize)(ptr - begin);
            return true;
        }
    }
    return false;
}

static bool checked_f32_bytes(uint64_t a, uint64_t b, uint64_t c,
                              uint64_t &bytes) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    uint64_t values = a * b;
    if (values != 0 && c > UINT64_MAX / values) return false;
    values *= c;
    if (values > UINT64_MAX / sizeof(float)) return false;
    bytes = values * sizeof(float);
    return true;
}

static bool checked_u64_product(uint64_t a, uint64_t b, uint64_t &product) {
    if (a != 0 && b > UINT64_MAX / a) return false;
    product = a * b;
    return true;
}

static bool shader_f32_domain(uint64_t a, uint64_t b, uint64_t c) {
    if (a != 0 && b > UINT32_MAX / a) return false;
    const uint64_t ab = a * b;
    return ab == 0 || c <= UINT32_MAX / ab;
}

/* Set only around the production full-layer router call whose routed
 * consumer is guaranteed to carry the deferred selected/weights barrier. */
static thread_local bool g_router_overlap_hint = false;

extern "C" void ds4_gpu_router_overlap_hint(int active) {
    g_router_overlap_hint = active != 0;
}

static int finish_simple_dispatch(VulkanCommandCtx &ctx, bool resume_recording,
                                  bool allow_router_defer = false) {
    const bool defer_router_dependency =
        allow_router_defer && g_router_overlap_hint && ctx.layer_batch_active;
    if (!defer_router_dependency) {
        VkMemoryBarrier barrier{};
        barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        timeline_barrier(ctx, "simple_compute_dependency");
        vkCmdPipelineBarrier(ctx.cmd,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             0, 1, &barrier, 0, nullptr, 0, nullptr);
    }
    ctx.command_count++;
    if (ctx.attention_output_batch) return 1;
    int ok = submit_and_wait();
    if (ok && resume_recording) ok = begin_cmd();
    return ok;
}

static int allocate_simple_descriptors(const ShaderEntry &shader,
                                       VkDescriptorBufferInfo *buffers,
                                       uint32_t count, VkDescriptorSet &set) {
    const VkDeviceSize alignment =
        (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if (alignment != 0) {
        for (uint32_t i = 0; i < count; i++)
            if (buffers[i].offset % alignment != 0) return 0;
    }
    VkDescriptorSetAllocateInfo allocate{};
    allocate.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocate.descriptorPool = g_vk.desc_pool;
    allocate.descriptorSetCount = 1;
    allocate.pSetLayouts = &shader.desc_layout;
    const uint64_t descriptor_start = timeline_now_ns();
    if (vkAllocateDescriptorSets(g_vk.device, &allocate, &set) != VK_SUCCESS)
        return 0;

    std::vector<VkWriteDescriptorSet> writes(count);
    for (uint32_t i = 0; i < count; i++) {
        writes[i] = {};
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = set;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &buffers[i];
    }
    vkUpdateDescriptorSets(g_vk.device, count, writes.data(), 0, nullptr);
    timeline_duration_current(TimelineEventKind::DescriptorAlloc,
                              "descriptor_cpu", descriptor_start);
    auto &ctx = get_cmd_ctx();
    mark_bound_weight_buffers(buffers, count, ctx.recording_generation);
    timeline_descriptors(ctx, shader.name.c_str(), buffers, count);
    return 1;
}

static int release_simple_descriptors(VkDescriptorSet set) {
    auto &ctx = get_cmd_ctx();
    if (ctx.layer_batch_active) {
        ctx.layer_batch_descriptors.push_back(set);
        maybe_submit();
        return 1;
    }
    const uint64_t descriptor_start = timeline_now_ns();
    const bool ok = vkFreeDescriptorSets(g_vk.device, g_vk.desc_pool, 1, &set) == VK_SUCCESS;
    if (ok) {
        timeline_resource(get_cmd_ctx(), TimelineEventKind::DescriptorFree,
                          "descriptor_set", 0);
        timeline_duration_current(TimelineEventKind::DescriptorFree,
                                  "descriptor_free_cpu", descriptor_start);
    }
    return ok;
}

static int release_or_defer_simple_descriptors(VulkanCommandCtx &ctx,
                                               VkDescriptorSet set) {
    if (ctx.attention_output_batch) {
        ctx.attention_output_descriptors.push_back(set);
        maybe_submit();
        return 1;
    }
    return release_simple_descriptors(set);
}

static void free_or_defer_attention_tensor(VulkanCommandCtx &ctx,
                                           ds4_gpu_tensor *tensor) {
    if (!tensor) return;
    if (ctx.attention_output_batch)
        ctx.attention_output_tensors.push_back(tensor);
    else
        ds4_gpu_tensor_free(tensor);
}

static bool checked_head_f32_layout(uint32_t n_tok, uint32_t n_head,
                                    uint32_t head_dim, uint64_t &row_bytes,
                                    uint64_t &total_bytes) {
    if (n_tok == 0 || n_head == 0 || head_dim == 0 ||
        (uint64_t)n_head > UINT64_MAX / head_dim)
        return false;
    const uint64_t row_elems = (uint64_t)n_head * head_dim;
    if (row_elems > UINT64_MAX / sizeof(float)) return false;
    row_bytes = row_elems * sizeof(float);
    if ((uint64_t)n_tok > UINT64_MAX / row_bytes) return false;
    total_bytes = (uint64_t)n_tok * row_bytes;
    return true;
}

static bool make_head_tile_descriptor(VkBuffer buffer, VkDeviceSize base_offset,
                                      uint64_t row_bytes, uint32_t token_base,
                                      uint32_t tile_tokens,
                                      VkDescriptorBufferInfo &info) {
    if (row_bytes == 0 || token_base > UINT64_MAX / row_bytes ||
        tile_tokens > UINT64_MAX / row_bytes)
        return false;
    const uint64_t tile_offset = (uint64_t)token_base * row_bytes;
    const uint64_t tile_bytes = (uint64_t)tile_tokens * row_bytes;
    if ((uint64_t)base_offset > UINT64_MAX - tile_offset ||
        (VkDeviceSize)((uint64_t)base_offset + tile_offset) !=
            (uint64_t)base_offset + tile_offset)
        return false;
    const VkDeviceSize offset = base_offset + (VkDeviceSize)tile_offset;
    const VkDeviceSize alignment =
        (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if (alignment != 0 && offset % alignment != 0) return false;
    info = {buffer, offset, (VkDeviceSize)tile_bytes};
    return true;
}

static int fail_simple_dispatch(VulkanCommandCtx &ctx) {
    if (ctx.recording && ctx.command_count == 0) (void)submit_and_wait();
    return 0;
}

static int record_simple_shader(const char *name, const void *push, uint32_t push_size,
                                VkDescriptorBufferInfo *buffers, uint32_t count,
                                uint32_t gx, uint32_t gy, uint32_t gz,
                                bool resume_recording) {
    auto si = g_vk.shader_map.find(name);
    if (si == g_vk.shader_map.end()) return 0;
    auto &ctx = get_cmd_ctx();
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, count, set)) return 0;
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    if (push_size)
        vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, push_size, push);
    timeline_dispatch(ctx, name, buffers, count, gx, gy, gz);
    ctx.command_count++;
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_or_defer_simple_descriptors(ctx, set)) ok = 0;
    return ok;
}

int ds4_gpu_rms_norm_plain_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                                   uint32_t n, float eps) {
    return ds4_gpu_rms_norm_plain_rows_tensor(out, x, n, 1, eps);
}

int ds4_gpu_rms_norm_plain_rows_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                                        uint32_t n, uint32_t rows, float eps) {
    DS4_VK_TRACE_KERNEL("rms_norm");
    if (!out || !x || n == 0 || rows == 0 ||
        (uint64_t)n * rows > x->bytes / sizeof(float) ||
        (uint64_t)n * rows > out->bytes / sizeof(float)) return 0;
    auto si = g_vk.shader_map.find("rms_norm");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer xbuf, obuf; VkDeviceSize xoff, ooff;
    if (!find_tensor_buffer(x, xbuf, xoff) || !find_tensor_buffer(out, obuf, ooff)) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo buffers[2] = {
        {xbuf, xoff, (VkDeviceSize)n * rows * sizeof(float)},
        {obuf, ooff, (VkDeviceSize)n * rows * sizeof(float)},
    };
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, 2, set))
        return fail_simple_dispatch(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    struct { uint32_t n, rows; float eps; } push = {n, rows, eps};
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    timeline_dispatch(ctx, "rms_norm_weight_rows", buffers, 3, rows, 1, 1);
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_simple_descriptors(set)) ok = 0;
    return ok;
}

int ds4_gpu_rms_norm_weight_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
                                    const void *mm, uint64_t ms, uint64_t woff,
                                    uint32_t n, float eps) {
    DS4_VK_TRACE_KERNEL("rms_norm_weight");
    return ds4_gpu_rms_norm_weight_rows_tensor(out, x, mm, ms, woff, n, 1, eps);
}

int ds4_gpu_swiglu_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *gate,
                           const ds4_gpu_tensor *up, uint32_t n, float clamp, float weight) {
    DS4_VK_TRACE_KERNEL("swiglu");
    if (!out || !gate || !up || n == 0 ||
        (uint64_t)n * sizeof(float) > gate->bytes ||
        (uint64_t)n * sizeof(float) > up->bytes ||
        (uint64_t)n * sizeof(float) > out->bytes) return 0;
    auto si = g_vk.shader_map.find("swiglu");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer gate_buf, up_buf, out_buf; VkDeviceSize gate_off, up_off, out_off;
    if (!find_tensor_buffer(gate, gate_buf, gate_off) ||
        !find_tensor_buffer(up, up_buf, up_off) ||
        !find_tensor_buffer(out, out_buf, out_off)) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo buffers[3] = {
        {gate_buf, gate_off, (VkDeviceSize)n * sizeof(float)},
        {up_buf, up_off, (VkDeviceSize)n * sizeof(float)},
        {out_buf, out_off, (VkDeviceSize)n * sizeof(float)},
    };
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, 3, set))
        return fail_simple_dispatch(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    struct { uint32_t n; float clamp; float weight; } push = {n, clamp, weight};
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    timeline_dispatch(ctx, "swiglu", buffers, 3, (n + 255) / 256, 1, 1);
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_simple_descriptors(set)) ok = 0;
    return ok;
}

int ds4_gpu_add_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *a, const ds4_gpu_tensor *b, uint32_t n) {
    DS4_VK_TRACE_KERNEL("add_f32");
    if (!out || !a || !b || n == 0 ||
        (uint64_t)n * sizeof(float) > a->bytes ||
        (uint64_t)n * sizeof(float) > b->bytes ||
        (uint64_t)n * sizeof(float) > out->bytes) return 0;
    auto si = g_vk.shader_map.find("add_f32");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer abuf, bbuf, obuf; VkDeviceSize aoff, boff, ooff;
    if (!find_tensor_buffer(a, abuf, aoff) || !find_tensor_buffer(b, bbuf, boff) ||
        !find_tensor_buffer(out, obuf, ooff)) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo buffers[3] = {
        {abuf, aoff, (VkDeviceSize)n * sizeof(float)},
        {bbuf, boff, (VkDeviceSize)n * sizeof(float)},
        {obuf, ooff, (VkDeviceSize)n * sizeof(float)},
    };
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, 3, set))
        return fail_simple_dispatch(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(n), &n);
    timeline_dispatch(ctx, "add_f32", buffers, 3, (n + 255) / 256, 1, 1);
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_simple_descriptors(set)) ok = 0;
    return ok;
}

int ds4_gpu_argmax_tensor(ds4_gpu_tensor *out_idx,
                          const ds4_gpu_tensor *logits,
                          uint32_t n_vocab) {
    if (!out_idx || !logits || !out_idx->ptr || !logits->ptr || n_vocab == 0 ||
        out_idx->bytes < sizeof(int32_t) ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float)) {
        return 0;
    }
    const float *values = (const float *)logits->ptr;
    uint32_t best = 0;
    float best_value = values[0];
    for (uint32_t i = 1; i < n_vocab; i++) {
        if (values[i] > best_value) {
            best = i;
            best_value = values[i];
        }
    }
    *(int32_t *)out_idx->ptr = (int32_t)best;
    return 1;
}

/* =========================================================================
 * Vulkan Compute Dispatch: matmul_q8_0
 * =========================================================================
 * Dispatches the GLSL matmul_q8_0 shader. Caches Q8_0 weight spans in
 * VkBuffer at first use. Uses thread-local descriptor set for efficiency.
 * ========================================================================= */

/* ---- matmul_q8_0 dispatch ---- */

int ds4_gpu_quantize_q8_0_tensor(
    ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
    uint64_t in_dim, uint64_t n_tok);
int ds4_gpu_matmul_q8_0_prequant_tensor(
    ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x_q8, uint64_t n_tok);

int ds4_gpu_matmul_q8_0_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok)
{
    DS4_VK_TRACE_KERNEL("matmul_q8_0");
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (in_dim > 8192u || out_dim > UINT32_MAX || n_tok > UINT32_MAX) return 0;
    uint64_t n_blocks = (in_dim + 31) / 32;
    if (n_blocks > UINT64_MAX / 34u || out_dim > UINT64_MAX / (n_blocks * 34u)) return 0;
    const uint64_t weight_bytes = (uint64_t)out_dim * n_blocks * 34u;
    if (weight_bytes > UINT32_MAX || (weight_bytes & 3u) != 0 ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        n_tok > UINT64_MAX / in_dim ||
        in_dim * n_tok > x->bytes / sizeof(float) ||
        n_tok > UINT64_MAX / out_dim ||
        out_dim * n_tok > out->bytes / sizeof(float)) return 0;

    const bool prequant_eligible =
        n_tok <= 65535u && n_blocks <= 256u &&
        n_tok <= UINT64_MAX / n_blocks &&
        n_tok * n_blocks <= UINT64_MAX / 36u &&
        n_tok * n_blocks * 36u <= UINT32_MAX &&
        g_vk.caps.max_compute_work_group_size[0] >= 256u &&
        g_vk.caps.max_compute_work_group_invocations >= 256u;
    if (prequant_eligible) {
        /* Quantized activation scratch is shader-produced and shader-consumed;
         * keeping it device-local removes a per-projection host-visible VMA
         * allocation while the existing synchronous/layer-ring lifetime rules
         * remain unchanged. */
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc_device_scratch(
            n_tok * n_blocks * 36u);
        if (!q) return 0;
        int ok = ds4_gpu_quantize_q8_0_tensor(q, x, in_dim, n_tok);
        if (ok)
            ok = ds4_gpu_matmul_q8_0_prequant_tensor(
                out, model_map, model_size, weight_offset, in_dim, out_dim, q, n_tok);
        auto &ctx = get_cmd_ctx();
        if (ctx.attention_output_batch) {
            free_or_defer_attention_tensor(ctx, q);
            return ok;
        }
        if (ok) {
            /* The temporary is referenced by the matmul command, so wait before freeing it. */
            ok = submit_and_wait();
        } else {
            if (ctx.recording && ctx.command_count != 0)
                submit_and_wait();
        }
        ds4_gpu_tensor_free(q);
        return ok;
    }

    /* Get shader */
    auto si = g_vk.shader_map.find("matmul_q8_0");
    if (si == g_vk.shader_map.end()) return 0;
    auto &sh = g_vk.shaders[si->second];
    auto &c = get_cmd_ctx();
    if (c.recording && c.command_count != 0 && !submit_and_wait()) return 0;
    if (!c.recording && !begin_cmd()) return 0;

    /* Find tensor buffers */
    auto find_buf = [](const void *ptr, VkBuffer &buf, VkDeviceSize &off) -> bool {
        auto it = g_vk.tensor_headers.find(const_cast<void*>(ptr));
        if (it != g_vk.tensor_headers.end()) { buf = it->second->buffer; off = 0; return true; }
        for (auto &[base, h] : g_vk.tensor_headers)
            if (ptr >= base && (const char*)ptr < (const char*)base + (int64_t)h->bytes) {
                buf = h->buffer; off = (const char*)ptr - (const char*)base; return true; }
        return false;
    };
    VkBuffer xbuf, obuf; VkDeviceSize xoff, ooff;
    if (!find_buf(x->ptr, xbuf, xoff) || !find_buf(out->ptr, obuf, ooff)) return 0;

    /* Lazy weight upload with LRU eviction: find the range in the cache or
     * upload it from the model mmap on first use. */
    VkBuffer wbuf; uint64_t wbuf_off;
    auto wit = g_vk.weight_cache.end();
    for (auto it = g_vk.weight_cache.begin(); it != g_vk.weight_cache.end(); ++it) {
        if (weight_offset >= it->first && weight_offset - it->first <= it->second.size &&
            weight_bytes <= it->second.size - (weight_offset - it->first)) {
            wit = it; break;
        }
    }
    if (wit != g_vk.weight_cache.end()) {
        wit->second.last_used = ++g_vk.lru_counter;
        mark_weight_generation(wit->second);
        wbuf = wit->second.buffer;
        wbuf_off = weight_offset - wit->first;
    } else {
        /* Q8_0 (GGUF) block = 34 bytes: f16 scale + 32 x int8, same as ds4.c. */
        if (!ensure_weight(weight_offset, weight_bytes)) return 0;
        wit = g_vk.weight_cache.find(weight_offset);
        if (wit == g_vk.weight_cache.end()) return 0;
        wbuf = wit->second.buffer;
        wbuf_off = 0;
    }

    const VkDeviceSize w_size = std::min<VkDeviceSize>(
        wit->second.size - (wbuf_off > wit->second.size ? 0 : wbuf_off),
        weight_bytes);
    const uint64_t tile_tokens = std::max<uint64_t>(1u, 32768u / out_dim);
    const VkDeviceSize storage_align =
        (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if (storage_align != 0 && ((VkDeviceSize)wbuf_off % storage_align) != 0)
        return 0;
    for (uint64_t tile = 0; tile < n_tok; tile += tile_tokens) {
        const uint32_t tile_n = (uint32_t)std::min<uint64_t>(tile_tokens, n_tok - tile);
        if ((uint64_t)tile_n * in_dim > UINT32_MAX ||
            (uint64_t)tile_n * out_dim > UINT32_MAX) return 0;
        if (c.recording && c.command_count != 0 && !submit_and_wait()) return 0;
        if (!c.recording && !begin_cmd()) return 0;
        VkDescriptorSetAllocateInfo dai{};
        dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        dai.descriptorPool = g_vk.desc_pool; dai.descriptorSetCount = 1;
        dai.pSetLayouts = &sh.desc_layout;
        VkDescriptorSet ds;
        if (vkAllocateDescriptorSets(g_vk.device, &dai, &ds) != VK_SUCCESS) return 0;
        const VkDeviceSize x_size = (VkDeviceSize)tile_n * in_dim * sizeof(float);
        const VkDeviceSize o_size = (VkDeviceSize)tile_n * out_dim * sizeof(float);
        const VkDeviceSize tile_x_off =
            xoff + (VkDeviceSize)tile * in_dim * sizeof(float);
        const VkDeviceSize tile_o_off =
            ooff + (VkDeviceSize)tile * out_dim * sizeof(float);
        if (storage_align != 0 &&
            ((tile_x_off % storage_align) != 0 ||
             (tile_o_off % storage_align) != 0)) {
            vkFreeDescriptorSets(g_vk.device, g_vk.desc_pool, 1, &ds);
            return 0;
        }
        VkDescriptorBufferInfo bufs[3] = {
            {xbuf, tile_x_off, x_size},
            {wbuf, wbuf_off, w_size},
            {obuf, tile_o_off, o_size},
        };
        VkWriteDescriptorSet w[3];
        for (int i = 0; i < 3; i++) {
            w[i] = {}; w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            w[i].dstSet = ds; w[i].dstBinding = i; w[i].descriptorCount = 1;
            w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bufs[i];
        }
        vkUpdateDescriptorSets(g_vk.device, 3, w, 0, nullptr);
        timeline_descriptors(c, "matmul_q8_0", bufs, 3);
        vkCmdBindPipeline(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sh.pipeline);
        vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sh.layout, 0, 1, &ds, 0, nullptr);
        const uint32_t y_scale = std::min((uint32_t)out_dim, 65534u);
        const uint32_t y_cnt = ((uint32_t)out_dim + y_scale - 1) / y_scale;
        struct { uint32_t in_dim, out_dim, n_tok, blocks, y_scale; } pc = {
            (uint32_t)in_dim, (uint32_t)out_dim, tile_n, (uint32_t)n_blocks, y_scale
        };
        vkCmdPushConstants(c.cmd, sh.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        timeline_dispatch(c, "matmul_q8_0", bufs, 3, y_scale, y_cnt, tile_n);
        c.command_count++;
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        timeline_barrier(c, "matmul_q8_0_compute_dependency");
        vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                             0, nullptr, 0, nullptr);
        if (!submit_and_wait()) return 0;
        if (!release_simple_descriptors(ds)) return 0;
        timeline_resource(c, TimelineEventKind::DescriptorFree,
                          "matmul_q8_0", 0);
    }
    return 1;
}

int ds4_gpu_quantize_q8_0_tensor(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
        uint64_t in_dim, uint64_t n_tok) {
    DS4_VK_TRACE_KERNEL("quantize_q8_0_prequant");
    if (!out || !x || in_dim == 0 || n_tok == 0 || in_dim > UINT32_MAX ||
        n_tok > 65535u) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (blocks > UINT32_MAX || n_tok > UINT64_MAX / blocks ||
        n_tok * blocks > UINT64_MAX / 36u ||
        n_tok * in_dim > x->bytes / sizeof(float) ||
        n_tok * blocks * 36u > out->bytes) return 0;

    VkBuffer xbuf, obuf; VkDeviceSize xoff, ooff;
    if (!find_tensor_buffer(x, xbuf, xoff) || !find_tensor_buffer(out, obuf, ooff)) return 0;
    const VkDeviceSize align = (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if ((align && ((xoff | ooff) % align) != 0) ||
        n_tok * in_dim * sizeof(float) > UINT32_MAX ||
        n_tok * blocks * 36u > UINT32_MAX ||
        (g_vk.caps.max_storage_buffer_range != 0 &&
         (n_tok * in_dim * sizeof(float) > g_vk.caps.max_storage_buffer_range ||
          n_tok * blocks * 36u > g_vk.caps.max_storage_buffer_range))) return 0;
    if (g_vk.caps.max_compute_work_group_size[0] < 32u ||
        g_vk.caps.max_compute_work_group_size[1] < 1u ||
        g_vk.caps.max_compute_work_group_size[2] < 1u ||
        g_vk.caps.max_compute_work_group_invocations < 32u ||
        blocks > g_vk.caps.max_compute_work_group_count[0] ||
        n_tok > g_vk.caps.max_compute_work_group_count[2]) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (!ctx.attention_output_batch && ctx.recording &&
        ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo buffers[2] = {
        {xbuf, xoff, (VkDeviceSize)(n_tok * in_dim * sizeof(float))},
        {obuf, ooff, (VkDeviceSize)(n_tok * blocks * 36u)}};
    struct { uint32_t in_dim, blocks_per_row, n_tok; } pc = {
        (uint32_t)in_dim, (uint32_t)blocks, (uint32_t)n_tok};
    return record_simple_shader("quantize_q8_0_prequant", &pc, sizeof(pc),
                                buffers, 2, (uint32_t)blocks, 1,
                                (uint32_t)n_tok, resume_recording);
}

int ds4_gpu_matmul_q8_0_prequant_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        const ds4_gpu_tensor *x_q8, uint64_t n_tok) {
    DS4_VK_TRACE_KERNEL("matmul_q8_0_prequant");
    if (!out || !x_q8 || !model_map || in_dim == 0 || out_dim == 0 ||
        n_tok == 0 || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        n_tok > 65535u) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (blocks > 256u || out_dim > UINT64_MAX / blocks ||
        out_dim * blocks > UINT64_MAX / 34u) return 0;
    const uint64_t weight_bytes = out_dim * blocks * 34u;
    const uint64_t q_bytes = n_tok * blocks * 36u;
    if (weight_bytes > UINT32_MAX || (weight_bytes & 3u) != 0 ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset ||
        q_bytes > x_q8->bytes || n_tok > UINT64_MAX / out_dim ||
        n_tok * out_dim > out->bytes / sizeof(float)) return 0;

    VkBuffer xbuf, obuf; VkDeviceSize xoff, ooff;
    if (!find_tensor_buffer(x_q8, xbuf, xoff) || !find_tensor_buffer(out, obuf, ooff)) return 0;
    const VkDeviceSize align = (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((xoff | ooff) % align) != 0) return 0;
    if (g_vk.caps.max_compute_work_group_size[0] < 256u ||
        g_vk.caps.max_compute_work_group_size[1] < 1u ||
        g_vk.caps.max_compute_work_group_size[2] < 1u ||
        g_vk.caps.max_compute_work_group_invocations < 256u) return 0;
    const uint32_t y_scale = std::min((uint32_t)out_dim,
                                      g_vk.caps.max_compute_work_group_count[0]);
    if (y_scale == 0) return 0;
    const uint64_t y_count64 = (out_dim + y_scale - 1u) / y_scale;
    if (y_count64 > g_vk.caps.max_compute_work_group_count[1] ||
        n_tok > g_vk.caps.max_compute_work_group_count[2]) return 0;
    const uint64_t output_bytes = n_tok * out_dim * sizeof(float);
    if (g_vk.caps.max_storage_buffer_range != 0 &&
        (q_bytes > g_vk.caps.max_storage_buffer_range ||
         output_bytes > g_vk.caps.max_storage_buffer_range)) return 0;
    if (g_vk.model_map != model_map || g_vk.model_size != model_size)
        set_model_map_identity(model_map, model_size);
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (!ctx.attention_output_batch && ctx.recording &&
        ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    const uint64_t requested_records = out_dim * blocks;
    const uint64_t requested_scale_bytes = (requested_records * 2u + 3u) & ~3ull;
    const uint64_t requested_payload_bytes = requested_records * 32u;
    decltype(g_vk.aligned_cache)::mapped_type *aligned = nullptr;
    bool use_aligned = ensure_aligned_weight(model_map, model_size, weight_offset,
                                              in_dim, out_dim, aligned);
    if (use_aligned &&
        (requested_scale_bytes > aligned->scale_bytes ||
         requested_payload_bytes > aligned->payload_bytes ||
         (g_vk.caps.max_storage_buffer_range != 0 &&
          (requested_scale_bytes > g_vk.caps.max_storage_buffer_range ||
           requested_payload_bytes > g_vk.caps.max_storage_buffer_range))))
        use_aligned = false;
    const char *q8_mode = getenv("DS4_VULKAN_Q8_MODE");
    const char *rows8_env = getenv("DS4_VULKAN_Q8_ROWS8");
    const char *rows2_env = getenv("DS4_VULKAN_Q8_ROWS2");
    const char *wave64_env = getenv("DS4_VULKAN_Q8_WAVE64");
    bool use_wave64 = use_aligned &&
        !(q8_mode && strcmp(q8_mode, "exact") == 0) &&
        !(wave64_env && strcmp(wave64_env, "0") == 0) &&
        g_vk.caps.subgroup_size == 64 && g_vk.caps.has_subgroup_shuffle &&
        n_tok == 1 && in_dim == 8192 && blocks == 256 && out_dim == 4096;
    bool use_rows2 = use_aligned &&
        !(q8_mode && strcmp(q8_mode, "exact") == 0) &&
        !(rows2_env && strcmp(rows2_env, "0") == 0) &&
        n_tok == 1 && in_dim == 4096 && blocks == 128 && out_dim == 1024;
    const bool use_rows8 = use_aligned &&
        !(q8_mode && strcmp(q8_mode, "exact") == 0) &&
        !(rows8_env && strcmp(rows8_env, "0") == 0) &&
        n_tok == 1 && in_dim == 1024 && blocks == 32 &&
        out_dim == 32768;
    const char *shader_name = use_wave64
        ? "matmul_q8_0_wave64_bfe"
        : (use_rows2
        ? "matmul_q8_0_rows2_bfe"
        : (use_rows8
        ? "matmul_q8_0_rows8_bfe"
        : (use_aligned
            ? (q8_mode && strcmp(q8_mode, "exact") == 0
                ? "matmul_q8_0_aligned" : "matmul_q8_0_aligned_bfe")
            : "matmul_q8_0_prequant")));
    auto si = g_vk.shader_map.find(shader_name);
    if (si == g_vk.shader_map.end() && use_wave64) {
        /* A stale shader bundle must preserve the aligned dispatch geometry. */
        use_wave64 = false;
        shader_name = use_rows2
            ? "matmul_q8_0_rows2_bfe"
            : (use_rows8 ? "matmul_q8_0_rows8_bfe" : "matmul_q8_0_aligned_bfe");
        si = g_vk.shader_map.find(shader_name);
    }
    if (si == g_vk.shader_map.end() && use_rows2) {
        /* A stale shader bundle must fall back to the matching one-row
         * dispatch geometry, never run the aligned shader with rows2's grid. */
        use_rows2 = false;
        shader_name = use_rows8
            ? "matmul_q8_0_rows8_bfe"
            : "matmul_q8_0_aligned_bfe";
        si = g_vk.shader_map.find(shader_name);
    }
    if (si == g_vk.shader_map.end() && use_rows8) {
        shader_name = "matmul_q8_0_aligned_bfe";
        si = g_vk.shader_map.find(shader_name);
    }
    if (si == g_vk.shader_map.end() && use_aligned) {
        use_aligned = false;
        shader_name = "matmul_q8_0_prequant";
        si = g_vk.shader_map.find(shader_name);
    }
    if (si == g_vk.shader_map.end()) return 0;
    auto &sh = g_vk.shaders[si->second];
    VkDescriptorBufferInfo buffers[4] = {};
    uint32_t descriptor_count = 0;
    buffers[descriptor_count++] = {xbuf, xoff, (VkDeviceSize)q_bytes};
    if (use_aligned) {
        buffers[descriptor_count++] = {aligned->gpu.buffer, 0,
                                       (VkDeviceSize)requested_scale_bytes};
        buffers[descriptor_count++] = {aligned->gpu.buffer,
                                       (VkDeviceSize)aligned->payload_offset,
                                       (VkDeviceSize)requested_payload_bytes};
    } else {
        VkBuffer weight_buffer = VK_NULL_HANDLE;
        VkDeviceSize weight_buffer_offset = 0;
        VkDeviceSize weight_range = 0;
        if (g_vk.caps.max_storage_buffer_range != 0 &&
            weight_bytes > g_vk.caps.max_storage_buffer_range) return 0;
        if (!find_model_buffer(weight_offset, weight_bytes, weight_buffer,
                               weight_buffer_offset, weight_range) ||
            (align && weight_buffer_offset % align != 0)) return 0;
        buffers[descriptor_count++] = {weight_buffer, weight_buffer_offset, weight_range};
    }
    buffers[descriptor_count++] = {obuf, ooff,
        (VkDeviceSize)(n_tok * out_dim * sizeof(float))};
    const uint32_t dispatch_x = use_wave64
        ? (uint32_t)out_dim
        : (use_rows2
        ? (uint32_t)((out_dim + 1u) / 2u)
        : (use_rows8 ? (uint32_t)((out_dim + 7u) / 8u) : y_scale));
    const uint32_t dispatch_y = (use_wave64 || use_rows2 || use_rows8)
        ? 1u : (uint32_t)y_count64;
    struct { uint32_t in_dim, out_dim, n_tok, blocks_per_row, y_scale; } pc = {
        (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok, (uint32_t)blocks,
        use_wave64 ? 1u : (use_rows2 ? 2u : (use_rows8 ? 8u : y_scale))};
    if (use_wave64 && getenv("DS4_VULKAN_TRACE_KERNELS"))
        fprintf(stderr,
                "ds4: [trace] matmul_q8_0_wave64_bfe shape=%ux%u blocks=%u "
                "dispatch=%ux%ux%u\n",
                (unsigned)in_dim, (unsigned)out_dim, (unsigned)blocks,
                (unsigned)dispatch_x, (unsigned)dispatch_y, (unsigned)n_tok);
    if (sh.push_size != sizeof(pc) ||
        g_vk.caps.max_push_constants_size < sizeof(pc)) return 0;
    return record_simple_shader(shader_name, &pc, sizeof(pc),
                                buffers, descriptor_count, dispatch_x, dispatch_y,
                                (uint32_t)n_tok, resume_recording);
}

/* Grouped decode appliance for contiguous Q8_0 output rows.  The input has
 * one prequantized row per group and the weights are laid out as
 * [group][out_row][block].  A single workgroup computes one output row; the
 * shader keeps the exact ascending block reduction used by the ordinary
 * aligned path. */
static int ds4_gpu_matmul_q8_0_group_tensor(
        ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
        uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
        uint32_t n_groups, const ds4_gpu_tensor *x_q8) {
    DS4_VK_TRACE_KERNEL("matmul_q8_0_group_bfe");
    if (!out || !x_q8 || !model_map || in_dim == 0 || out_dim == 0 ||
        n_groups == 0 || in_dim > UINT32_MAX || out_dim > UINT32_MAX ||
        !out->ptr || !x_q8->ptr) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (blocks == 0 || blocks > 256u ||
        (uint64_t)n_groups > UINT64_MAX / blocks ||
        (uint64_t)n_groups * blocks > UINT64_MAX / 36u ||
        (uint64_t)n_groups * blocks * 36u > x_q8->bytes ||
        (uint64_t)n_groups > UINT64_MAX / out_dim ||
        (uint64_t)n_groups * out_dim > UINT64_MAX / (blocks * 34u) ||
        (uint64_t)n_groups * out_dim * sizeof(float) > out->bytes)
        return 0;
    const uint64_t total_rows = (uint64_t)n_groups * out_dim;
    const uint64_t weight_bytes = total_rows * blocks * 34u;
    if (weight_offset > model_size || weight_bytes > model_size - weight_offset)
        return 0;
    /* Pack as many rows as fit in a 256-lane workgroup.  The appliance
     * shader keeps one lane per (row, Q8 block), so this is exact whenever
     * rows_per_workgroup * blocks <= 256.  Keep the old one-row shader as a
     * clean fallback for stale bundles or unusual devices. */
    const uint32_t rows_per_workgroup = std::max<uint32_t>(
        1u, std::min<uint32_t>(256u, 256u / (uint32_t)blocks));
    const uint64_t packed_dispatch_x =
        ((uint64_t)out_dim + rows_per_workgroup - 1u) / rows_per_workgroup;
    if (packed_dispatch_x > g_vk.caps.max_compute_work_group_count[0] ||
        n_groups > g_vk.caps.max_compute_work_group_count[1]) return 0;
    auto rows_si = g_vk.shader_map.find("matmul_q8_0_group_rows_bfe");
    auto one_si = g_vk.shader_map.find("matmul_q8_0_group_bfe");
    const bool use_packed = rows_si != g_vk.shader_map.end();
    auto si = use_packed ? rows_si : one_si;
    if (si == g_vk.shader_map.end()) return 0;

    decltype(g_vk.aligned_cache)::mapped_type *aligned = nullptr;
    if (!ensure_aligned_weight(model_map, model_size, weight_offset,
                               in_dim, total_rows, aligned) || !aligned)
        return 0;
    const uint64_t records = total_rows * blocks;
    const uint64_t scale_bytes = (records * 2u + 3u) & ~3ull;
    const uint64_t payload_bytes = records * 32u;
    if (scale_bytes > aligned->scale_bytes ||
        payload_bytes > aligned->payload_bytes ||
        (g_vk.caps.max_storage_buffer_range != 0 &&
         (scale_bytes > g_vk.caps.max_storage_buffer_range ||
          payload_bytes > g_vk.caps.max_storage_buffer_range))) return 0;

    VkBuffer xbuf, obuf; VkDeviceSize xoff, ooff;
    if (!find_tensor_buffer(x_q8, xbuf, xoff) ||
        !find_tensor_buffer(out, obuf, ooff)) return 0;
    const VkDeviceSize align = (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((xoff | ooff) % align) != 0) return 0;
    const VkDeviceSize out_bytes = (VkDeviceSize)total_rows * sizeof(float);
    if (g_vk.caps.max_storage_buffer_range != 0 &&
        (n_groups * blocks * 36u > g_vk.caps.max_storage_buffer_range ||
         out_bytes > g_vk.caps.max_storage_buffer_range)) return 0;

    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (!ctx.attention_output_batch && ctx.recording &&
        ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo buffers[4] = {
        {xbuf, xoff, (VkDeviceSize)n_groups * blocks * 36u},
        {aligned->gpu.buffer, 0, (VkDeviceSize)scale_bytes},
        {aligned->gpu.buffer, (VkDeviceSize)aligned->payload_offset,
         (VkDeviceSize)payload_bytes},
        {obuf, ooff, out_bytes},
    };
    struct { uint32_t in_dim, out_dim, n_groups, blocks_per_row, rows_per_workgroup; } pc = {
        (uint32_t)in_dim, (uint32_t)out_dim, n_groups, (uint32_t)blocks,
        use_packed ? rows_per_workgroup : 1u};
    const char *shader_name = use_packed ?
        "matmul_q8_0_group_rows_bfe" : "matmul_q8_0_group_bfe";
    return record_simple_shader(shader_name, &pc, sizeof(pc),
                                buffers, 4,
                                use_packed ? (uint32_t)packed_dispatch_x : (uint32_t)out_dim,
                                n_groups, 1u, resume_recording);
}

/* ---- matmul_f32_tensor dispatch (host-side f32 matmul) ---- */
int ds4_gpu_matmul_f32_tensor(
    ds4_gpu_tensor *out, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x, uint64_t n_tok)
{
    (void)model_map; (void)model_size;
    if (!out || !out->ptr || !x || !x->ptr || !model_map) return 0;
    if (out->bytes < n_tok * out_dim * sizeof(float) ||
        x->bytes < n_tok * in_dim * sizeof(float)) return 0;
    const float *xp = (const float *)x->ptr;
    const float *wp = (const float *)((const char *)model_map + weight_offset);
    float *op = (float *)out->ptr;
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t o = 0; o < out_dim; o++) {
            double acc = 0.0;
            for (uint64_t i = 0; i < in_dim; i++)
                acc += (double)xp[t * in_dim + i] * (double)wp[o * in_dim + i];
            op[t * out_dim + o] = (float)acc;
        }
    }
    return 1;
}

/* ---- matmul_f16_tensor dispatch ---- */
int ds4_gpu_matmul_f16_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok)
{
    DS4_VK_TRACE_KERNEL("matmul_f16");
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0 ||
        in_dim > UINT32_MAX || out_dim > UINT32_MAX || n_tok > UINT32_MAX) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr, "ds4: [dbg] matmul_f16 invalid arguments\n");
        return 0;
    }
    if (out_dim > UINT64_MAX / in_dim || out_dim * in_dim > UINT64_MAX / 2u ||
        weight_offset > model_size || out_dim * in_dim * 2u > model_size - weight_offset ||
        n_tok > UINT64_MAX / in_dim || n_tok * in_dim > x->bytes / sizeof(float) ||
        n_tok > UINT64_MAX / out_dim || n_tok * out_dim > out->bytes / sizeof(float))
        return 0;

    const char *f16_mode = getenv("DS4_VULKAN_F16_MODE");
    const char *shader_name = f16_mode && strcmp(f16_mode, "exact") == 0
        ? "matmul_f16" : "matmul_f16_fast";
    auto si = g_vk.shader_map.find(shader_name);
    if (si == g_vk.shader_map.end()) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr, "ds4: [dbg] matmul_f16 shader unavailable\n");
        return 0;
    }
    auto &sh = g_vk.shaders[si->second];
    auto &c = get_cmd_ctx();
    if (!c.recording && !begin_cmd()) return 0;

    auto find_buf = [](const void *ptr, VkBuffer &buf, VkDeviceSize &off) -> bool {
        auto it = g_vk.tensor_headers.find(const_cast<void*>(ptr));
        if (it != g_vk.tensor_headers.end()) { buf = it->second->buffer; off = 0; return true; }
        for (auto &[base, h] : g_vk.tensor_headers)
            if (ptr >= base && (const char*)ptr < (const char*)base + (int64_t)h->bytes) {
                buf = h->buffer; off = (const char*)ptr - (const char*)base; return true; }
        return false;
    };
    VkBuffer xbuf, obuf; VkDeviceSize xoff, ooff;
    if (!find_buf(x->ptr, xbuf, xoff) || !find_buf(out->ptr, obuf, ooff)) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr,
                    "ds4: [dbg] matmul_f16 tensor buffer lookup failed "
                    "in=%llu out=%llu tok=%llu x=%p dst=%p\n",
                    (unsigned long long)in_dim,
                    (unsigned long long)out_dim,
                    (unsigned long long)n_tok,
                    x->ptr,
                    out->ptr);
        return 0;
    }
    /* Lazy weight upload with LRU eviction: find the range in the cache or
     * upload it from the model mmap on first use. */
    const uint64_t weight_bytes = (uint64_t)out_dim * in_dim * 2u;
    VkBuffer wbuf; uint64_t wbuf_off;
    auto wit = g_vk.weight_cache.end();
    for (auto it = g_vk.weight_cache.begin(); it != g_vk.weight_cache.end(); ++it) {
        if (weight_offset >= it->first && weight_offset - it->first <= it->second.size &&
            weight_bytes <= it->second.size - (weight_offset - it->first)) {
            wit = it; break;
        }
    }
    if (wit != g_vk.weight_cache.end()) {
        wit->second.last_used = ++g_vk.lru_counter;
        mark_weight_generation(wit->second);
        wbuf = wit->second.buffer;
        wbuf_off = weight_offset - wit->first;
    } else {
        /* W_f16 is a half-precision matrix: 2 bytes per element. */
        if (!ensure_weight(weight_offset, weight_bytes)) {
            if (getenv("DS4_VULKAN_DEBUG"))
                fprintf(stderr,
                        "ds4: [dbg] matmul_f16 weight upload failed "
                        "offset=%llu bytes=%llu used=%llu budget=%llu gen=%llu\n",
                        (unsigned long long)weight_offset,
                        (unsigned long long)((uint64_t)out_dim * in_dim * 2u),
                        (unsigned long long)g_vk.weight_used,
                        (unsigned long long)g_vk.weight_budget,
                        (unsigned long long)g_vk.cmd_gen);
            return 0;
        }
        wit = g_vk.weight_cache.find(weight_offset);
        if (wit == g_vk.weight_cache.end()) {
            if (getenv("DS4_VULKAN_DEBUG"))
                fprintf(stderr,
                        "ds4: [dbg] matmul_f16 uploaded weight missing from cache "
                        "offset=%llu\n",
                        (unsigned long long)weight_offset);
            return 0;
        }
        wbuf = wit->second.buffer;
        wbuf_off = 0;
    }
    const VkDeviceSize w_size = std::min<VkDeviceSize>(
        wit->second.size - (wbuf_off > wit->second.size ? 0 : wbuf_off),
        weight_bytes);  /* f16 weights: 2 bytes per element */
    if (out_dim > g_vk.caps.max_compute_work_group_count[0] ||
        sh.push_size != 3u * sizeof(uint32_t)) return 0;
    constexpr uint64_t max_tokens_per_dispatch = 256u;
    const uint64_t x_row_bytes = in_dim * sizeof(float);
    const uint64_t out_row_bytes = out_dim * sizeof(float);
    const VkDeviceSize align =
        (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    for (uint64_t token_base = 0; token_base < n_tok; ) {
        const uint32_t tile_tokens = (uint32_t)std::min<uint64_t>(
            max_tokens_per_dispatch, n_tok - token_base);
        if (!c.recording && !begin_cmd()) return 0;
        if (token_base > UINT64_MAX / x_row_bytes ||
            token_base > UINT64_MAX / out_row_bytes) return fail_simple_dispatch(c);
        const uint64_t x_delta = token_base * x_row_bytes;
        const uint64_t out_delta = token_base * out_row_bytes;
        if (x_delta > UINT64_MAX - xoff || out_delta > UINT64_MAX - ooff)
            return fail_simple_dispatch(c);
        const VkDeviceSize tile_xoff = xoff + x_delta;
        const VkDeviceSize tile_ooff = ooff + out_delta;
        if ((align && ((tile_xoff | tile_ooff | (VkDeviceSize)wbuf_off) % align) != 0))
            return fail_simple_dispatch(c);
        VkDescriptorBufferInfo bufs[3] = {
            {xbuf, tile_xoff, (VkDeviceSize)tile_tokens * x_row_bytes},
            {wbuf, (VkDeviceSize)wbuf_off, w_size},
            {obuf, tile_ooff, (VkDeviceSize)tile_tokens * out_row_bytes},
        };
        struct { uint32_t in_dim, out_dim, n_tok; } pc = {
            (uint32_t)in_dim, (uint32_t)out_dim, tile_tokens};
        const bool more_tiles = token_base + tile_tokens < n_tok;
        if (!record_simple_shader(shader_name, &pc, sizeof(pc), bufs, 3,
                                  (uint32_t)out_dim, tile_tokens, 1,
                                  more_tiles)) return 0;
        token_base += tile_tokens;
    }
    return 1;
}

/* ---- rms_norm_weight_rows_tensor dispatch ---- */
int ds4_gpu_rms_norm_weight_rows_tensor(
    ds4_gpu_tensor *out, const ds4_gpu_tensor *x,
    const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint32_t n, uint32_t rows, float eps)
{
    DS4_VK_TRACE_KERNEL("rms_norm_weight_rows");
    const uint64_t values = (uint64_t)n * rows;
    const uint64_t weight_bytes = (uint64_t)n * sizeof(float);
    if (!out || !x || !model_map || n == 0 || rows == 0 ||
        values > x->bytes / sizeof(float) || values > out->bytes / sizeof(float) ||
        weight_offset > model_size || weight_bytes > model_size - weight_offset) return 0;
    if (g_vk.model_map != model_map || g_vk.model_size != model_size)
        set_model_map_identity(model_map, model_size);
    auto si = g_vk.shader_map.find("rms_norm_weight_rows");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer xbuf, obuf; VkDeviceSize xoff, ooff;
    if (!find_tensor_buffer(x, xbuf, xoff) || !find_tensor_buffer(out, obuf, ooff)) return 0;
    if (!ensure_weight(weight_offset, weight_bytes)) return 0;

    VkBuffer wbuf = VK_NULL_HANDLE;
    VkDeviceSize woff = 0;
    auto wit = g_vk.weight_cache.end();
    for (auto it = g_vk.weight_cache.begin(); it != g_vk.weight_cache.end(); ++it) {
        if (weight_offset >= it->first &&
            weight_offset - it->first <= it->second.size &&
            weight_bytes <= it->second.size - (weight_offset - it->first)) {
            wit = it;
            break;
        }
    }
    if (wit == g_vk.weight_cache.end()) return 0;
    wit->second.last_used = ++g_vk.lru_counter;
    mark_weight_generation(wit->second);
    wbuf = wit->second.buffer;
    woff = weight_offset - wit->first;
    const VkDeviceSize alignment = (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if (alignment != 0 && (woff % alignment) != 0) return 0;

    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo buffers[3] = {
        {xbuf, xoff, (VkDeviceSize)values * sizeof(float)},
        {wbuf, woff, (VkDeviceSize)weight_bytes},
        {obuf, ooff, (VkDeviceSize)values * sizeof(float)},
    };
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, 3, set))
        return fail_simple_dispatch(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    struct { uint32_t n, rows; float eps; } push = {n, rows, eps};
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    timeline_dispatch(ctx, "rms_norm_weight_rows", buffers, 3, rows, 1, 1);
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_simple_descriptors(set)) ok = 0;
    return ok;
}

extern "C" {
/* ---- head_rms_norm_rope_tail_tensor ---- */
int ds4_gpu_head_rms_norm_rope_tail_tensor(ds4_gpu_tensor *x,
    uint32_t n_tok, uint32_t n_head, uint32_t head_dim, uint32_t n_rot,
    uint32_t pos0, uint32_t n_ctx_orig, bool inverse, float freq_base,
    float freq_scale, float ext_factor, float attn_factor, float beta_fast,
    float beta_slow, float eps)
{
    DS4_VK_TRACE_KERNEL("head_rms_norm_rope_tail");
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 || freq_base <= 0.0f ||
        freq_scale <= 0.0f) return 0;
    uint64_t row_bytes, total_bytes;
    if (n_head > 65535u ||
        !checked_head_f32_layout(n_tok, n_head, head_dim, row_bytes, total_bytes) ||
        total_bytes > x->bytes ||
        (n_tok > 1 && pos0 > UINT32_MAX - (n_tok - 1u))) return 0;
    auto si = g_vk.shader_map.find("head_rms_norm_rope_tail");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer buffer; VkDeviceSize offset;
    if (!find_tensor_buffer(x, buffer, offset)) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    /* Match the standalone RoPE work budget: 16K head rows each rotate 32
     * pairs for the production shape. Larger logical prefills remain one
     * engine chunk but are emitted as independently synchronized GPU tiles. */
    constexpr uint32_t max_head_rows_per_dispatch = 16384u;
    const uint32_t tile_tokens = std::max(
        1u, std::min(65535u, max_head_rows_per_dispatch) / n_head);
    const VkDeviceSize alignment =
        (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if (n_tok > tile_tokens && alignment != 0 && row_bytes % alignment != 0)
        return 0;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    auto &shader = g_vk.shaders[si->second];
    int ok = 1;
    for (uint32_t token_base = 0; token_base < n_tok; ) {
        const uint32_t tile_n = std::min(tile_tokens, n_tok - token_base);
        if (!ctx.recording && !begin_cmd()) return 0;
        VkDescriptorBufferInfo info;
        if (!make_head_tile_descriptor(buffer, offset, row_bytes, token_base,
                                       tile_n, info)) return fail_simple_dispatch(ctx);
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (!allocate_simple_descriptors(shader, &info, 1, set))
            return fail_simple_dispatch(ctx);
        vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
        vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                shader.layout, 0, 1, &set, 0, nullptr);
        struct Push {
            uint32_t n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig;
            int32_t inverse;
            float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps;
        } push = {tile_n, n_head, head_dim, n_rot, pos0 + token_base, n_ctx_orig,
                  inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, eps};
        vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        timeline_dispatch(ctx, "head_rms_norm_rope_tail", &info, 1,
                  tile_n * n_head, 1, 1);
        ok = finish_simple_dispatch(ctx, resume_recording);
        if (!release_simple_descriptors(set)) ok = 0;
        if (!ok) return 0;
        token_base += tile_n;
    }
    return ok;
}

/* ---- head_rms_norm_tensor dispatch ---- */
int ds4_gpu_head_rms_norm_tensor(ds4_gpu_tensor *x, uint32_t n_tok,
                                  uint32_t n_head, uint32_t head_dim, float eps)
{
    DS4_VK_TRACE_KERNEL("head_rms_norm");
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0) return 0;
    uint64_t row_bytes, total_bytes;
    if (n_head > 65535u ||
        !checked_head_f32_layout(n_tok, n_head, head_dim, row_bytes, total_bytes) ||
        total_bytes > x->bytes) return 0;
    auto si = g_vk.shader_map.find("head_rms_norm");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer buffer; VkDeviceSize offset;
    if (!find_tensor_buffer(x, buffer, offset)) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    const uint32_t tile_tokens = std::max(1u, 65535u / n_head);
    const VkDeviceSize alignment =
        (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if (n_tok > tile_tokens && alignment != 0 && row_bytes % alignment != 0)
        return 0;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    auto &shader = g_vk.shaders[si->second];
    int ok = 1;
    for (uint32_t token_base = 0; token_base < n_tok; ) {
        const uint32_t tile_n = std::min(tile_tokens, n_tok - token_base);
        if (!ctx.recording && !begin_cmd()) return 0;
        VkDescriptorBufferInfo info;
        if (!make_head_tile_descriptor(buffer, offset, row_bytes, token_base,
                                       tile_n, info)) return fail_simple_dispatch(ctx);
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (!allocate_simple_descriptors(shader, &info, 1, set))
            return fail_simple_dispatch(ctx);
        vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
        vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                shader.layout, 0, 1, &set, 0, nullptr);
        struct Push { uint32_t n_tok, n_head, head_dim; float eps; } push =
            {tile_n, n_head, head_dim, eps};
        vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        timeline_dispatch(ctx, "head_rms_norm", &info, 1,
                  tile_n * n_head, 1, 1);
        ok = finish_simple_dispatch(ctx, resume_recording);
        if (!release_simple_descriptors(set)) ok = 0;
        if (!ok) return 0;
        token_base += tile_n;
    }
    return ok;
}

/* ---- rope_tail_tensor dispatch ---- */
/* Matches ds4.c rope_tail_ext_inplace exactly:
 *   - RoPE touches only the tail of each head: the last n_rot of head_dim
 *     channels; the first head_dim - n_rot channels stay untouched.
 *   - Angles are built by accumulating theta_extrap *= theta_scale with
 *     theta_scale = freq_base^(-2/n_rot) (NOT head_dim), so the frequency
 *     of the i-th pair is freq_base^(-2*i/n_rot) * pos.
 *   - YaRN (corr_dims + ramp mix + magnitude scaling) is applied when
 *     ext_factor != 0.
 *   - The rotation is scaled by attn_factor, and inverse mode flips the
 *     sign of the sine term (conjugate rotation). */
int ds4_gpu_rope_tail_tensor(ds4_gpu_tensor *x, uint32_t n_tok, uint32_t n_head,
                              uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
                              uint32_t n_ctx_orig, bool inverse, float freq_base,
                              float freq_scale, float ext_factor, float attn_factor,
                              float beta_fast, float beta_slow)
{
    DS4_VK_TRACE_KERNEL("rope_tail");
    if (!x || n_tok == 0 || n_head == 0 || head_dim == 0 || n_rot == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 || freq_base <= 0.0f ||
        freq_scale <= 0.0f) return 0;
    uint64_t row_bytes, total_bytes;
    if (!checked_head_f32_layout(n_tok, n_head, head_dim, row_bytes, total_bytes) ||
        total_bytes > x->bytes ||
        (n_tok > 1 && pos0 > UINT32_MAX - (n_tok - 1u))) return 0;
    auto si = g_vk.shader_map.find("rope_tail");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer buffer; VkDeviceSize offset;
    if (!find_tensor_buffer(x, buffer, offset)) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    auto &shader = g_vk.shaders[si->second];
    const uint64_t pairs_per_token = (uint64_t)n_head * (n_rot / 2u);
    if (pairs_per_token == 0 || pairs_per_token > UINT64_MAX / 256u) return 0;
    /* RoPE evaluates several transcendentals per pair. Bound each dispatch to
     * the work exercised by a 256-token indexer batch, while leaving the
     * engine's logical prefill chunk and KV boundaries unchanged. */
    constexpr uint64_t max_groups_per_dispatch = 2048u;
    constexpr uint64_t invocations_per_group = 256u;
    const uint64_t pairs_per_dispatch =
        max_groups_per_dispatch * invocations_per_group;
    const uint32_t tile_tokens = (uint32_t)std::min<uint64_t>(n_tok,
        pairs_per_dispatch / pairs_per_token);
    if (tile_tokens == 0) return 0;
    const VkDeviceSize alignment =
        (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if (n_tok > tile_tokens && alignment != 0 && row_bytes % alignment != 0)
        return 0;
    int ok = 1;
    for (uint32_t token_base = 0; token_base < n_tok; ) {
        const uint32_t tile_n = std::min(tile_tokens, n_tok - token_base);
        if (!ctx.recording && !begin_cmd()) return 0;
        VkDescriptorBufferInfo info;
        if (!make_head_tile_descriptor(buffer, offset, row_bytes, token_base,
                                       tile_n, info)) return fail_simple_dispatch(ctx);
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (!allocate_simple_descriptors(shader, &info, 1, set))
            return fail_simple_dispatch(ctx);
        vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
        vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                                shader.layout, 0, 1, &set, 0, nullptr);
        struct Push {
            uint32_t n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig;
            int32_t inverse;
            float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
        } push = {tile_n, n_head, head_dim, n_rot, pos0 + token_base, n_ctx_orig,
                  inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow};
        vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        const uint32_t groups = tile_n;
        if (groups == 0 || groups > 65535u) {
            if (!release_simple_descriptors(set)) return 0;
            return fail_simple_dispatch(ctx);
        }
        timeline_dispatch(ctx, "rope_tail", &info, 1, groups, 1, 1);
        ok = finish_simple_dispatch(ctx, resume_recording);
        if (!release_simple_descriptors(set)) ok = 0;
        if (!ok) return 0;
        token_base += tile_n;
    }
    return ok;
}

static float ds4_half_to_float(uint16_t h);
static void ds4_embed_f16_row(float *out, const uint8_t *row, uint64_t n_embd);
static int ds4_embed_row_ok(const void *model_map, uint64_t model_size,
                            uint64_t weight_offset, uint64_t id, uint64_t row_bytes);

/* ---- embed_token_hc_tensor ---- */
int ds4_gpu_embed_token_hc_tensor(ds4_gpu_tensor *out_hc,
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint32_t n_vocab, uint32_t token, uint32_t n_embd, uint32_t n_hc)
{
    DS4_VK_TRACE_KERNEL("embed_token_hc");
    if (!out_hc || !model_map || n_vocab == 0 || n_embd == 0 || n_hc == 0) return 0;
    int32_t id = (int32_t)token;
    if (id < 0) id = 0;
    if ((uint32_t)id >= n_vocab) id = 0;
    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t out_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    if (out_hc->bytes < out_bytes ||
        !ds4_embed_row_ok(model_map, model_size, weight_offset, (uint64_t)id, row_bytes)) {
        return 0;
    }
    float *out = (float *)out_hc->ptr;
    ds4_embed_f16_row(out,
                      (const uint8_t *)model_map + weight_offset + (uint64_t)id * row_bytes,
                      n_embd);
    for (uint32_t h = 1; h < n_hc; h++) {
        memcpy(out + (uint64_t)h * n_embd, out, (size_t)n_embd * sizeof(float));
    }
    return 1;
}

/* ---- embed_tokens_hc_tensor ---- */
int ds4_gpu_embed_tokens_hc_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *tokens,
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd, uint32_t n_hc)
{
    if (!out_hc || !tokens || !model_map || n_vocab == 0 ||
        n_tokens == 0 || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t out_bytes = (uint64_t)n_tokens * n_hc * n_embd * sizeof(float);
    if (tokens->bytes < (uint64_t)n_tokens * sizeof(int32_t) || out_hc->bytes < out_bytes)
        return 0;
    const int32_t *tok = (const int32_t*)tokens->ptr;
    float *out = (float*)out_hc->ptr;
    for (uint32_t t = 0; t < n_tokens; t++) {
        int32_t id = tok[t];
        if (id < 0) id = 0;
        if ((uint32_t)id >= n_vocab) id = 0;
        if (!ds4_embed_row_ok(model_map, model_size, weight_offset,
                              (uint64_t)id, row_bytes)) return 0;
        float *token_out = out + (uint64_t)t * n_hc * n_embd;
        ds4_embed_f16_row(token_out,
                          (const uint8_t *)model_map + weight_offset +
                              (uint64_t)id * row_bytes,
                          n_embd);
        for (uint32_t h = 1; h < n_hc; h++) {
            memcpy(token_out + (uint64_t)h * n_embd,
                   token_out, (size_t)n_embd * sizeof(float));
        }
    }
    return 1;
}

/* ---- Q8_0 / F16 token embeddings (host-side dequant) ----
 *
 * Same pattern as the *_hc_tensor embedders above, but the model table is
 * quantized: Q8_0 rows are blocks of {scale f16, 32 x int8} (34 bytes per
 * block, GGUF layout), F16 rows are plain IEEE halves.  Every row is
 * dequantized to f32 directly into the output tensor; tokens outside
 * [0, n_vocab) clamp to row 0 (matches the HC embedders).  Rows are never
 * read past model_size (mmap SIGBUS guard).
 */
/* Dequantize one Q8_0 embedding row into out[0..n_embd).  The last block
 * may be partial when n_embd % 32 != 0; its tail int8s are ignored
 * (matches ds4.c embed_token_q8_0). */
static void ds4_embed_q8_0_row(float *out, const uint8_t *row, uint64_t n_embd) {
    const uint64_t blocks = (n_embd + 31u) / 32u;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
        const float scale = ds4_half_to_float(scale_bits);
        const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
        const uint64_t i0 = b * 32u;
        const uint64_t bn = n_embd - i0 < 32u ? n_embd - i0 : 32u;
        for (uint64_t i = 0; i < bn; i++) out[i0 + i] = scale * (float)qs[i];
    }
}

/* Decode one F16 embedding row into out[0..n_embd). */
static void ds4_embed_f16_row(float *out, const uint8_t *row, uint64_t n_embd) {
    for (uint64_t i = 0; i < n_embd; i++) {
        uint16_t bits;
        memcpy(&bits, row + i * 2u, sizeof(bits));
        out[i] = ds4_half_to_float(bits);
    }
}

/* True when the row [weight_offset + id*row_bytes,
 * weight_offset + (id+1)*row_bytes) is fully inside the model map. */
static int ds4_embed_row_ok(const void *model_map, uint64_t model_size,
                            uint64_t weight_offset, uint64_t id, uint64_t row_bytes) {
    if (!model_map || weight_offset >= model_size) return 0;
    const uint64_t avail = model_size - weight_offset;
    if (id > avail / row_bytes) return 0;
    return row_bytes <= avail - id * row_bytes;
}

/* ---- embed_token_q8_0_tensor ---- */
int ds4_gpu_embed_token_q8_0_tensor(ds4_gpu_tensor *out,
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint32_t n_vocab, uint32_t token, uint32_t n_embd)
{
    if (!out) return 0;
    int32_t id = (int32_t)token;
    if (id < 0 || (uint32_t)id >= n_vocab) id = 0;
    const uint64_t row_bytes = ((uint64_t)n_embd + 31u) / 32u * 34u;
    if (!ds4_embed_row_ok(model_map, model_size, weight_offset, (uint64_t)id, row_bytes))
        return 0;
    ds4_embed_q8_0_row((float *)out->ptr,
                       (const uint8_t *)model_map + weight_offset + (uint64_t)id * row_bytes,
                       n_embd);
    return 1;
}

/* ---- embed_tokens_q8_0_tensor ---- */
int ds4_gpu_embed_tokens_q8_0_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens,
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd)
{
    if (!out || !tokens) return 0;
    const int32_t *tok = (const int32_t *)tokens->ptr;
    float *outf = (float *)out->ptr;
    const uint64_t row_bytes = ((uint64_t)n_embd + 31u) / 32u * 34u;
    for (uint32_t t = 0; t < n_tokens; t++) {
        int32_t id = tok[t];
        if (id < 0 || (uint32_t)id >= n_vocab) id = 0;
        if (!ds4_embed_row_ok(model_map, model_size, weight_offset, (uint64_t)id, row_bytes))
            return 0;
        ds4_embed_q8_0_row(outf + (uint64_t)t * n_embd,
                           (const uint8_t *)model_map + weight_offset + (uint64_t)id * row_bytes,
                           n_embd);
    }
    return 1;
}

/* ---- embed_token_f16_tensor (static; dispatch target) ---- */
static int ds4_embed_token_f16_tensor(ds4_gpu_tensor *out,
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint32_t n_vocab, uint32_t token, uint32_t n_embd)
{
    if (!out) return 0;
    int32_t id = (int32_t)token;
    if (id < 0 || (uint32_t)id >= n_vocab) id = 0;
    const uint64_t row_bytes = (uint64_t)n_embd * 2u;
    if (!ds4_embed_row_ok(model_map, model_size, weight_offset, (uint64_t)id, row_bytes))
        return 0;
    ds4_embed_f16_row((float *)out->ptr,
                      (const uint8_t *)model_map + weight_offset + (uint64_t)id * row_bytes,
                      n_embd);
    return 1;
}

/* ---- embed_tokens_f16_tensor (static; dispatch target) ---- */
static int ds4_embed_tokens_f16_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *tokens,
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint32_t n_vocab, uint32_t n_tokens, uint32_t n_embd)
{
    if (!out || !tokens) return 0;
    const int32_t *tok = (const int32_t *)tokens->ptr;
    float *outf = (float *)out->ptr;
    const uint64_t row_bytes = (uint64_t)n_embd * 2u;
    for (uint32_t t = 0; t < n_tokens; t++) {
        int32_t id = tok[t];
        if (id < 0 || (uint32_t)id >= n_vocab) id = 0;
        if (!ds4_embed_row_ok(model_map, model_size, weight_offset, (uint64_t)id, row_bytes))
            return 0;
        ds4_embed_f16_row(outf + (uint64_t)t * n_embd,
                          (const uint8_t *)model_map + weight_offset + (uint64_t)id * row_bytes,
                          n_embd);
    }
    return 1;
}

/* ---- embed_token_quant_tensor (dispatch on weight_type) ---- */
int ds4_gpu_embed_token_quant_tensor(ds4_gpu_tensor *out,
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint32_t weight_type, uint32_t n_vocab, uint32_t token, uint32_t n_embd)
{
    switch (weight_type) {
    case 8:  /* DS4_TENSOR_Q8_0 */
        return ds4_gpu_embed_token_q8_0_tensor(out, model_map, model_size,
                                               weight_offset, n_vocab, token, n_embd);
    case 1:  /* DS4_TENSOR_F16 */
        return ds4_embed_token_f16_tensor(out, model_map, model_size,
                                          weight_offset, n_vocab, token, n_embd);
    default:
        return 0;  /* unsupported weight type */
    }
}

/* ---- embed_tokens_quant_tensor (dispatch on weight_type) ---- */
int ds4_gpu_embed_tokens_quant_tensor(ds4_gpu_tensor *out,
    const ds4_gpu_tensor *tokens, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint32_t weight_type, uint32_t n_vocab,
    uint32_t n_tokens, uint32_t n_embd)
{
    switch (weight_type) {
    case 8:  /* DS4_TENSOR_Q8_0 */
        return ds4_gpu_embed_tokens_q8_0_tensor(out, tokens, model_map, model_size,
                                                weight_offset, n_vocab, n_tokens, n_embd);
    case 1:  /* DS4_TENSOR_F16 */
        return ds4_embed_tokens_f16_tensor(out, tokens, model_map, model_size,
                                           weight_offset, n_vocab, n_tokens, n_embd);
    default:
        return 0;  /* unsupported weight type */
    }
}

/* ---- attn_q_b_f16_head_rms_rope_tail_tensor ---- */
int ds4_gpu_attn_q_b_f16_head_rms_rope_tail_tensor(
    ds4_gpu_tensor *q, ds4_gpu_tensor *q_half,
    const void *model_map, uint64_t model_size, uint64_t w_off, uint64_t in_dim,
    uint64_t out_dim, const ds4_gpu_tensor *qr_norm, uint32_t n_tok,
    uint32_t n_head, uint32_t head_dim, uint32_t n_rot, uint32_t pos0,
    uint32_t n_ctx_orig, bool inverse, float freq_base, float freq_scale,
    float ext_factor, float attn_factor, float beta_fast, float beta_slow,
    float eps)
{
    (void)q_half; (void)model_size; (void)w_off; (void)in_dim; (void)out_dim;
    (void)qr_norm; (void)n_ctx_orig; (void)inverse;
    (void)ext_factor; (void)attn_factor; (void)beta_fast; (void)beta_slow;
    /* Unavailable: the caller runs matmul_q8_0, head RMSNorm, and RoPE. */
    return 0;
}

} /* extern "C" */

/* ---- IEEE-754 half helpers (ds4.c f32_to_f16 / f16_to_f32) ----
 * The engine's raw KV cache stores rows through an f16 round trip
 * (kv_cache_push_raw, CUDA store_raw_kv_batch_kernel), so the Vulkan
 * store path must reproduce the same rounding to keep decode attention
 * bit-compatible with the other backends.
 */
static uint16_t ds4_float_to_half(float f) {
    uint32_t bits;
    memcpy(&bits, &f, sizeof(bits));

    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t exp = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;

    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        const uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half_mant = mant >> shift;
        const uint32_t round_bit = (mant >> (shift - 1)) & 1u;
        const uint32_t sticky = mant & ((1u << (shift - 1)) - 1u);
        if (round_bit && (sticky || (half_mant & 1u))) half_mant++;
        return (uint16_t)(sign | half_mant);
    }

    if (exp >= 31) {
        if (((bits >> 23) & 0xffu) == 0xffu && mant != 0) {
            return (uint16_t)(sign | 0x7e00u);
        }
        return (uint16_t)(sign | 0x7c00u);
    }

    uint32_t half = sign | ((uint32_t)exp << 10) | (mant >> 13);
    const uint32_t round = mant & 0x1fffu;
    if (round > 0x1000u || (round == 0x1000u && (half & 1u))) half++;
    return (uint16_t)half;
}

static float ds4_half_to_float(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t expo = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (expo == 0) {
        if (mant == 0) {
            bits = sign;                              /* +/- zero */
        } else {                                      /* subnormal */
            int e = -14;
            uint32_t m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; e--; }
            bits = sign | (uint32_t)(e + 127) << 23 | (m & 0x3ffu) << 13;
        }
    } else if (expo == 31) {
        bits = sign | 0x7f800000u | (mant << 13);     /* inf / nan */
    } else {
        bits = sign | (expo + 112u) << 23 | mant << 13;
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

/* ---- DSV4-specific extern C implementations ---- */
extern "C" {

int ds4_gpu_dsv4_fp8_kv_quantize_tensor(
    ds4_gpu_tensor *x,
    uint32_t n_tok,
    uint32_t head_dim,
    uint32_t n_rot)
{
    uint64_t tensor_bytes;
    if (!x || !x->ptr || n_tok == 0 || n_tok > 65535u ||
        head_dim == 0 || n_rot > head_dim ||
        !shader_f32_domain(n_tok, 1, head_dim) ||
        !checked_f32_bytes(n_tok, 1, head_dim, tensor_bytes) ||
        x->bytes < tensor_bytes)
        return 0;
    VkBuffer xbuf; VkDeviceSize xoff;
    if (!find_tensor_buffer(x, xbuf, xoff) ||
        (g_vk.caps.min_storage_buffer_offset_alignment != 0 &&
         xoff % g_vk.caps.min_storage_buffer_offset_alignment != 0)) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[1] = {{xbuf, xoff, (VkDeviceSize)x->bytes}};
    struct { uint32_t n_tok, head_dim, n_rot; } pc = {n_tok, head_dim, n_rot};
    DS4_VK_TRACE_KERNEL("fp8_kv_quantize");
    return record_simple_shader("fp8_kv_quantize", &pc, sizeof(pc), bufs, 1,
                                n_tok, 1, 1, resume_recording);
}

int ds4_gpu_attention_prefill_raw_heads_tensor(
    ds4_gpu_tensor *heads,
    const void *model_map,
    uint64_t model_size,
    uint64_t sinks_offset,
    const ds4_gpu_tensor *q,
    const ds4_gpu_tensor *raw_kv,
    uint32_t n_tokens,
    uint32_t window,
    uint32_t n_head,
    uint32_t head_dim)
{
    uint64_t head_bytes, raw_bytes;
    if (!heads || !q || !raw_kv || !model_map || n_tokens == 0 || n_head == 0 ||
        head_dim == 0 || window == 0 || window > 256 ||
        n_tokens > 65535u || n_head > 65535u ||
        !shader_f32_domain(n_tokens, n_head, head_dim) ||
        !shader_f32_domain(n_tokens, 1, head_dim) ||
        !checked_f32_bytes(n_tokens, n_head, head_dim, head_bytes) ||
        !checked_f32_bytes(n_tokens, 1, head_dim, raw_bytes) ||
        heads->bytes < head_bytes || q->bytes < head_bytes ||
        raw_kv->bytes < raw_bytes ||
        sinks_offset > model_size || (uint64_t)n_head * sizeof(float) > model_size - sinks_offset)
        return 0;
    VkBuffer obuf, qbuf, kvbuf, sbuf; VkDeviceSize ooff, qoff, kvoff, soff, ssize;
    if (!find_tensor_buffer(heads, obuf, ooff) || !find_tensor_buffer(q, qbuf, qoff) ||
        !find_tensor_buffer(raw_kv, kvbuf, kvoff) ||
        !find_model_buffer(sinks_offset, (uint64_t)n_head * sizeof(float), sbuf, soff, ssize)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((ooff | qoff | kvoff | soff) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[4] = {{obuf, ooff, (VkDeviceSize)heads->bytes},
                                      {qbuf, qoff, (VkDeviceSize)q->bytes},
                                      {kvbuf, kvoff, (VkDeviceSize)raw_kv->bytes},
                                      {sbuf, soff, ssize}};
    struct { uint32_t n_tokens, window, n_head, head_dim; } pc = {n_tokens, window, n_head, head_dim};
    DS4_VK_TRACE_KERNEL("attention_prefill_raw");
    return record_simple_shader("attention_prefill_raw", &pc, sizeof(pc), bufs, 4,
                                n_tokens, n_head, 1, resume_recording);
}

int ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
    ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q,
    const void *mm, uint64_t ms, uint64_t qwo, uint32_t qn,
    ds4_gpu_tensor *kv_out, const ds4_gpu_tensor *kv,
    uint64_t kvwo, uint32_t kvn, uint32_t rows, float eps)
{
    if (!q_out || !q || !kv_out || !kv || !mm) return 0;
    uint64_t q_bytes, kv_bytes;
    if (qn == 0 || kvn == 0 || rows == 0 || rows > 65535u ||
        !checked_f32_bytes(rows, 1, qn, q_bytes) ||
        !checked_f32_bytes(rows, 1, kvn, kv_bytes)) return 0;
    if (qwo > ms || (uint64_t)qn * sizeof(float) > ms - qwo ||
        kvwo > ms || (uint64_t)kvn * sizeof(float) > ms - kvwo ||
        q_out->bytes < q_bytes || q->bytes < q_bytes ||
        kv_out->bytes < kv_bytes || kv->bytes < kv_bytes)
        return 0;

    return ds4_gpu_rms_norm_weight_rows_tensor(
               q_out, q, mm, ms, qwo, qn, rows, eps) &&
           ds4_gpu_rms_norm_weight_rows_tensor(
               kv_out, kv, mm, ms, kvwo, kvn, rows, eps);
}

/* Compose the Vulkan Q/KV RMSNorm dispatches with the Vulkan RoPE dispatch. */
int ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor(
    ds4_gpu_tensor *q_out, const ds4_gpu_tensor *q,
    const void *mm, uint64_t ms, uint64_t qwo, uint32_t qn,
    ds4_gpu_tensor *kv_out, const ds4_gpu_tensor *kv,
    uint64_t kvwo, uint32_t kvn, uint32_t rows,
    uint32_t kv_n_head, uint32_t kv_head_dim, uint32_t n_rot,
    uint32_t pos0, uint32_t n_ctx_orig, bool inverse,
    float freq_base, float freq_scale, float ext_factor,
    float attn_factor, float beta_fast, float beta_slow, float eps)
{
    if (!q_out || !q || !kv_out || !kv || !mm) return 0;
    if (qn == 0 || kvn == 0 || rows == 0) return 0;
    if (kv_n_head == 0 || kv_head_dim == 0 ||
        n_rot > kv_head_dim || (n_rot & 1u) ||
        (uint64_t)kvn != (uint64_t)kv_n_head * kv_head_dim)
        return 0;

    if (!ds4_gpu_dsv4_qkv_rms_norm_rows_tensor(
            q_out, q, mm, ms, qwo, qn, kv_out, kv, kvwo, kvn, rows, eps))
        return 0;
    return ds4_gpu_rope_tail_tensor(
        kv_out, rows, kv_n_head, kv_head_dim, n_rot,
        pos0, n_ctx_orig, inverse, freq_base, freq_scale,
        ext_factor, attn_factor, beta_fast, beta_slow);
}

extern "C" int ds4_gpu_kv_rope_fp8_fuse_available(void) { return 0; }

/* Compose Vulkan FP8 quantization with the Vulkan raw-cache row store. */
int ds4_gpu_kv_fp8_store_raw_tensor(
    ds4_gpu_tensor *kv,
    ds4_gpu_tensor *raw_cache,
    uint32_t          raw_cap,
    uint32_t          row,
    uint32_t          head_dim,
    uint32_t          n_rot)
{
    if (!kv || !kv->ptr || !raw_cache || !raw_cache->ptr) return 0;
    if (raw_cap == 0 || head_dim == 0 || n_rot > head_dim) return 0;
    uint64_t row_bytes, cache_bytes;
    if (!shader_f32_domain(raw_cap, 1, head_dim) ||
        !checked_f32_bytes(1, 1, head_dim, row_bytes) ||
        !checked_f32_bytes(raw_cap, 1, head_dim, cache_bytes) ||
        kv->bytes < row_bytes || raw_cache->bytes < cache_bytes)
        return 0;

    return ds4_gpu_dsv4_fp8_kv_quantize_tensor(kv, 1, head_dim, n_rot) &&
           ds4_gpu_store_raw_kv_batch_tensor(raw_cache, kv, raw_cap, row, 1,
                                              head_dim);
}

} /* extern "C" DSV4 implementations */

/* ---- Remaining critical functions ---- */
extern "C" {

int ds4_gpu_store_raw_kv_batch_tensor(ds4_gpu_tensor *raw_cache, const ds4_gpu_tensor *kv,
    uint32_t raw_cap, uint32_t pos0, uint32_t n_tokens, uint32_t head_dim)
{
    uint64_t cache_bytes, kv_bytes;
    if (!raw_cache || !kv || raw_cap == 0 || n_tokens == 0 || head_dim == 0 ||
        (n_tokens > 1 && pos0 > UINT32_MAX - (n_tokens - 1u)) ||
        !shader_f32_domain(raw_cap, 1, head_dim) ||
        !shader_f32_domain(n_tokens, 1, head_dim) ||
        !checked_f32_bytes(raw_cap, 1, head_dim, cache_bytes) ||
        !checked_f32_bytes(n_tokens, 1, head_dim, kv_bytes) ||
        raw_cache->bytes < cache_bytes || kv->bytes < kv_bytes) return 0;
    VkBuffer rbuf, kbuf; VkDeviceSize roff, koff;
    if (!find_tensor_buffer(raw_cache, rbuf, roff) || !find_tensor_buffer(kv, kbuf, koff)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((roff | koff) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[2] = {{rbuf, roff, (VkDeviceSize)raw_cache->bytes},
                                      {kbuf, koff, (VkDeviceSize)kv->bytes}};
    struct { uint32_t raw_cap, pos0, n_tokens, head_dim; } pc = {raw_cap, pos0, n_tokens, head_dim};
    const uint64_t groups = ((uint64_t)n_tokens * head_dim + 255u) / 256u;
    if (groups == 0 || groups > 65535u) return fail_simple_dispatch(ctx);
    DS4_VK_TRACE_KERNEL("store_raw_kv_f16");
    return record_simple_shader("store_raw_kv_f16", &pc, sizeof(pc), bufs, 2,
                                (uint32_t)groups,
                                1, 1, resume_recording);
}

/* Vulkan can keep the inverse RoPE tail inside the head-owned mixed-attention
 * workgroup.  ds4.c arms this narrowly for the ordinary (non-indexed)
 * decode path and checks the consumed flag before skipping its standalone
 * rope dispatch. */
struct VulkanDecodeAttnRopeFuse {
    bool armed = false;
    bool used = false;
    uint32_t n_rot = 0;
    uint32_t pos0 = 0;
    uint32_t n_ctx_orig = 0;
    bool inverse = false;
    float freq_base = 0.0f;
    float freq_scale = 0.0f;
    float ext_factor = 0.0f;
    float attn_factor = 1.0f;
    float beta_fast = 0.0f;
    float beta_slow = 0.0f;
};
static thread_local VulkanDecodeAttnRopeFuse g_decode_attn_rope_fuse;

extern "C" int ds4_gpu_decode_attn_rope_fuse_available(void) {
    const char *wave64_env = getenv("DS4_VULKAN_ATTN_WAVE64");
    return g_vk.caps.subgroup_size == 64u &&
        !(wave64_env && strcmp(wave64_env, "0") == 0) &&
        g_vk.shader_map.find("attention_decode_mixed_rope") !=
            g_vk.shader_map.end();
}

extern "C" int ds4_gpu_decode_attn_rope_fuse_used(void) {
    return g_decode_attn_rope_fuse.used ? 1 : 0;
}

extern "C" void ds4_gpu_set_decode_attn_rope_fuse(
        uint32_t n_head_dim, uint32_t n_rot, uint32_t pos0,
        uint32_t n_ctx_orig, bool inverse, float freq_base,
        float freq_scale, float ext_factor, float attn_factor,
        float beta_fast, float beta_slow) {
    (void)n_head_dim;
    g_decode_attn_rope_fuse.armed = true;
    g_decode_attn_rope_fuse.used = false;
    g_decode_attn_rope_fuse.n_rot = n_rot;
    g_decode_attn_rope_fuse.pos0 = pos0;
    g_decode_attn_rope_fuse.n_ctx_orig = n_ctx_orig;
    g_decode_attn_rope_fuse.inverse = inverse;
    g_decode_attn_rope_fuse.freq_base = freq_base;
    g_decode_attn_rope_fuse.freq_scale = freq_scale;
    g_decode_attn_rope_fuse.ext_factor = ext_factor;
    g_decode_attn_rope_fuse.attn_factor = attn_factor;
    g_decode_attn_rope_fuse.beta_fast = beta_fast;
    g_decode_attn_rope_fuse.beta_slow = beta_slow;
}

/* ---- ds4_gpu_attention_decode_heads_tensor ----
 *
 * Single-token causal decode attention over the raw ring cache plus the
 * compressed (MLA) cache.  The operation is recorded and synchronized as
 * a Vulkan compute dispatch.
 *
 * Replicates the engine's CPU reference (ds4.c layer_attention_mixed_one)
 * and the CUDA attention_decode_mixed_kernel math exactly:
 *   scale = 1/sqrtf(head_dim)
 *   per head h: qh = q[h*head_dim ..] (f32); sink prior sinks[h]
 *     (f32 at model_map + sinks_offset, n_head values) joins the softmax
 *     denominator only.
 *   raw rows are the n_raw newest chronological rows of the ring:
 *     row r (r = 0..n_raw-1) lives at (raw_start + r) % raw_cap and holds
 *     head_dim f32 values.  K and V share the row (MLA-style single KV per
 *     token); every q head attends to the same rows.
 *   compressed rows c = 0..n_comp-1 live at comp_kv[c*head_dim ..]; the
 *     storage is f32 or IEEE f16 (comp_kv_f16, 2 bytes/element) and the
 *     comp_mask (used when use_mask != 0) is an additive float bias per
 *     row: 0.0 = allowed, -inf = masked (rows with bias <= -1e20 are
  *     excluded, matching the CUDA kernel).
  *     score = dot(qh, kv) * scale (+ comp mask bias); softmax with the sink
  *     prior; heads[h*head_dim ..] = sum(exp(score - max) * v) / denom.
  */
int ds4_gpu_attention_decode_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        uint32_t                n_raw,
        uint32_t                raw_cap,
        uint32_t                raw_start,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_comp,
        const ds4_gpu_tensor *comp_mask,
        uint32_t                use_mask,
        uint32_t                n_head,
        uint32_t                head_dim)
{
    uint64_t head_bytes, raw_bytes;
    const uint64_t comp_values = (uint64_t)n_comp * head_dim;
    if (!heads || !q || !raw_kv || !model_map || n_raw == 0 || n_head == 0 || head_dim == 0 ||
        raw_cap < n_raw || raw_start >= raw_cap || (n_comp && !comp_kv) ||
        n_raw > 1024u || n_comp > 1024u - n_raw ||
        (use_mask && !comp_mask) || sinks_offset > model_size ||
        (uint64_t)n_head * sizeof(float) > model_size - sinks_offset ||
        n_head > 65535u ||
        !shader_f32_domain(1, n_head, head_dim) ||
        !shader_f32_domain(raw_cap, 1, head_dim) ||
        !shader_f32_domain(n_comp, 1, head_dim) ||
        !checked_f32_bytes(1, n_head, head_dim, head_bytes) ||
        !checked_f32_bytes(raw_cap, 1, head_dim, raw_bytes) ||
        heads->bytes < head_bytes || q->bytes < head_bytes || raw_kv->bytes < raw_bytes ||
        (comp_kv_f16 && (comp_values & 1u) != 0) ||
        (n_comp && (comp_values > UINT64_MAX / (comp_kv_f16 ? 2u : 4u) ||
                comp_kv->bytes < comp_values * (comp_kv_f16 ? 2u : 4u))) ||
        (use_mask && comp_mask->bytes < (uint64_t)n_comp * sizeof(float))) return 0;
    VkBuffer obuf, qbuf, rbuf, cbuf, mbuf, sbuf;
    VkDeviceSize ooff, qoff, roff, coff, moff, soff, ssize;
    if (!find_tensor_buffer(heads, obuf, ooff) || !find_tensor_buffer(q, qbuf, qoff) ||
        !find_tensor_buffer(raw_kv, rbuf, roff) ||
        !find_model_buffer(sinks_offset, (uint64_t)n_head * sizeof(float), sbuf, soff, ssize)) return 0;
    if (n_comp) {
        if (!find_tensor_buffer(comp_kv, cbuf, coff)) return 0;
    } else { cbuf = rbuf; coff = roff; }
    if (use_mask) {
        if (!find_tensor_buffer(comp_mask, mbuf, moff)) return 0;
    } else { mbuf = rbuf; moff = roff; }
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((ooff | qoff | roff | coff | moff | soff) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    const bool use_fused_rope =
        g_decode_attn_rope_fuse.armed &&
        g_decode_attn_rope_fuse.n_rot != 0 &&
        g_decode_attn_rope_fuse.n_rot <= head_dim &&
        (g_decode_attn_rope_fuse.n_rot & 1u) == 0 &&
        ds4_gpu_decode_attn_rope_fuse_available() != 0;
    VkDescriptorBufferInfo bufs[6] = {
        {obuf, ooff, (VkDeviceSize)heads->bytes}, {qbuf, qoff, (VkDeviceSize)q->bytes},
        {rbuf, roff, (VkDeviceSize)raw_kv->bytes},
        {cbuf, coff, n_comp ? (VkDeviceSize)comp_kv->bytes : 4},
        {mbuf, moff, use_mask ? (VkDeviceSize)comp_mask->bytes : 4}, {sbuf, soff, ssize}
    };
    struct Push {
        uint32_t n_raw, raw_cap, raw_start, n_comp, comp_f16, use_mask, n_head, head_dim;
        uint32_t rope_enable, n_rot, pos0, n_ctx_orig;
        int32_t inverse;
        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    } pc = {n_raw, raw_cap, raw_start, n_comp, comp_kv_f16, use_mask, n_head, head_dim,
            use_fused_rope ? 1u : 0u,
            use_fused_rope ? g_decode_attn_rope_fuse.n_rot : 0u,
            use_fused_rope ? g_decode_attn_rope_fuse.pos0 : 0u,
            use_fused_rope ? g_decode_attn_rope_fuse.n_ctx_orig : 0u,
            use_fused_rope && g_decode_attn_rope_fuse.inverse ? 1 : 0,
            use_fused_rope ? g_decode_attn_rope_fuse.freq_base : 0.0f,
            use_fused_rope ? g_decode_attn_rope_fuse.freq_scale : 1.0f,
            use_fused_rope ? g_decode_attn_rope_fuse.ext_factor : 0.0f,
            use_fused_rope ? g_decode_attn_rope_fuse.attn_factor : 1.0f,
            use_fused_rope ? g_decode_attn_rope_fuse.beta_fast : 0.0f,
            use_fused_rope ? g_decode_attn_rope_fuse.beta_slow : 0.0f};
    const char *wave64_env = getenv("DS4_VULKAN_ATTN_WAVE64");
    const bool use_wave64 = g_vk.caps.subgroup_size == 64u &&
        !(wave64_env && strcmp(wave64_env, "0") == 0) &&
        g_vk.shader_map.find("attention_decode_mixed_wave64") !=
            g_vk.shader_map.end();
    const char *rope512_env = getenv("DS4_VULKAN_ATTN_DECODE_ROPE_WAVE64_512");
    const bool use_fused_rope_wave64_512 = use_fused_rope && head_dim == 512u &&
        g_decode_attn_rope_fuse.n_rot == 64u &&
        g_vk.caps.subgroup_size == 64u &&
        !(rope512_env && strcmp(rope512_env, "0") == 0) &&
        g_vk.shader_map.find("attention_decode_mixed_rope_wave64_512") !=
            g_vk.shader_map.end();
    const char *shader_name = use_fused_rope_wave64_512 ?
        "attention_decode_mixed_rope_wave64_512" :
        (use_fused_rope ? "attention_decode_mixed_rope" :
        (use_wave64 ? "attention_decode_mixed_wave64" :
                      "attention_decode_mixed"));
    DS4_VK_TRACE_KERNEL(shader_name);
    int ok = record_simple_shader(shader_name, &pc,
                                  use_fused_rope ? sizeof(pc) : 32u,
                                  bufs, 6, n_head, 1, 1, resume_recording);
    if (use_fused_rope && ok) {
        g_decode_attn_rope_fuse.armed = false;
        g_decode_attn_rope_fuse.used = true;
    }
    return ok;
}

int ds4_gpu_attention_decode_raw_batch_heads_tensor(
    ds4_gpu_tensor *heads, const void *mm, uint64_t ms, uint64_t sinks_off,
    const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv, uint32_t n_tokens,
    uint32_t pos0, uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
    uint32_t window, uint32_t n_head, uint32_t head_dim)
{
    uint64_t head_bytes, raw_bytes;
    if (!heads || !q || !raw_kv || !mm || n_tokens == 0 || n_raw == 0 ||
        raw_cap == 0 || n_raw > raw_cap || raw_start >= raw_cap ||
        window == 0 || window > 256 || n_head == 0 || head_dim == 0 ||
        pos0 > UINT32_MAX - n_tokens ||
        (uint64_t)n_raw > (uint64_t)pos0 + n_tokens ||
        n_tokens > 65535u || n_head > 65535u ||
        !shader_f32_domain(n_tokens, n_head, head_dim) ||
        !shader_f32_domain(raw_cap, 1, head_dim) ||
        !checked_f32_bytes(n_tokens, n_head, head_dim, head_bytes) ||
        !checked_f32_bytes(raw_cap, 1, head_dim, raw_bytes) ||
        heads->bytes < head_bytes || q->bytes < head_bytes ||
        raw_kv->bytes < raw_bytes ||
        sinks_off > ms || (uint64_t)n_head * sizeof(float) > ms - sinks_off)
        return 0;
    VkBuffer obuf, qbuf, rbuf, sbuf;
    VkDeviceSize ooff, qoff, roff, soff, ssize;
    if (!find_tensor_buffer(heads, obuf, ooff) || !find_tensor_buffer(q, qbuf, qoff) ||
        !find_tensor_buffer(raw_kv, rbuf, roff) ||
        !find_model_buffer(sinks_off, (uint64_t)n_head * sizeof(float), sbuf, soff, ssize))
        return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((ooff | qoff | roff | soff) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[4] = {
        {obuf, ooff, (VkDeviceSize)heads->bytes},
        {qbuf, qoff, (VkDeviceSize)q->bytes},
        {rbuf, roff, (VkDeviceSize)raw_kv->bytes},
        {sbuf, soff, ssize},
    };
    struct { uint32_t n_tokens, pos0, n_raw, raw_cap, raw_start, window, n_head, head_dim; }
        pc = {n_tokens, pos0, n_raw, raw_cap, raw_start, window, n_head, head_dim};
    DS4_VK_TRACE_KERNEL("attention_decode_raw_batch");
    return record_simple_shader("attention_decode_raw_batch", &pc, sizeof(pc), bufs, 4,
                                n_tokens, n_head, 1, resume_recording);
}

/* ---- attention output projections ----
 *
 * Grouped attention output (DeepSeek V4 Flash).  The n_groups attention
 * groups each own group_dim inputs (the concatenated heads of the group)
 * and are projected to a rank-dimensional low vector by the Q8_0 matrix
 * out_a; the concatenated low vector (n_groups*rank) is then projected to
 * out_dim by the Q8_0 matrix out_b.
 *
 * The GPU composition preserves ds4.c's grouped layout: gather one group's
 * strided activation rows, run the existing Q8_0 matmul for out_a, scatter
 * the rank rows into low, then run the same Q8_0 matmul for out_b.
 * Q8_0 rows are GGUF blocks of {f16 scale, 32 x int8} = 34 bytes per block.
 * The existing matmul shader performs the activation-side Q8_0 quantization.
 */

static int ds4_vk_group_copy(const ds4_gpu_tensor *src, ds4_gpu_tensor *dst,
                             uint32_t width, uint32_t src_stride,
                             uint32_t dst_stride, uint32_t rows) {
    if (!src || !dst || width == 0 || rows == 0 ||
    rows > 65535u || !shader_f32_domain(rows, 1, src_stride) ||
    !shader_f32_domain(rows, 1, dst_stride) ||
        src_stride < width || dst_stride < width ||
        (uint64_t)(rows - 1u) * src_stride + width > src->bytes / sizeof(float) ||
        (uint64_t)(rows - 1u) * dst_stride + width > dst->bytes / sizeof(float))
        return 0;
    auto si = g_vk.shader_map.find("group_copy");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer sbuf, dbuf; VkDeviceSize soff, doff;
    if (!find_tensor_buffer(src, sbuf, soff) || !find_tensor_buffer(dst, dbuf, doff)) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (!ctx.attention_output_batch && ctx.recording &&
        ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    const VkDeviceSize sbytes = ((VkDeviceSize)(rows - 1u) * src_stride + width) * sizeof(float);
    const VkDeviceSize dbytes = ((VkDeviceSize)(rows - 1u) * dst_stride + width) * sizeof(float);
    const VkDeviceSize alignment = g_vk.caps.min_storage_buffer_offset_alignment;
    const VkDeviceSize saligned = alignment ? soff - soff % alignment : soff;
    const VkDeviceSize daligned = alignment ? doff - doff % alignment : doff;
    const VkDeviceSize sdelta = soff - saligned;
    const VkDeviceSize ddelta = doff - daligned;
    if (sdelta % sizeof(float) != 0 || ddelta % sizeof(float) != 0 ||
        sdelta / sizeof(float) > UINT32_MAX || ddelta / sizeof(float) > UINT32_MAX)
        return fail_simple_dispatch(ctx);
    VkDescriptorBufferInfo buffers[2] = {
        {sbuf, saligned, sdelta + sbytes}, {dbuf, daligned, ddelta + dbytes}};
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, 2, set)) return fail_simple_dispatch(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.layout,
                            0, 1, &set, 0, nullptr);
    struct {
        uint32_t width, src_stride, dst_stride, rows;
        uint32_t src_offset, dst_offset;
    } pc = {width, src_stride, dst_stride, rows,
            (uint32_t)(sdelta / sizeof(float)),
            (uint32_t)(ddelta / sizeof(float))};
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    timeline_dispatch(ctx, "group_copy", buffers, 2,
                      (width + 255u) / 256u, rows, 1);
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_or_defer_simple_descriptors(ctx, set)) ok = 0;
    return ok;
}

static int ds4_vk_attention_output_low_gpu(
        ds4_gpu_tensor *low, const void *model_map, uint64_t model_size,
        uint64_t out_a_offset, uint64_t group_dim, uint64_t rank,
        uint32_t n_groups, const ds4_gpu_tensor *heads, uint32_t n_tokens) {
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups == 0 || n_tokens == 0 || group_dim > UINT32_MAX || rank > UINT32_MAX ||
        n_tokens > UINT32_MAX || n_groups > UINT32_MAX) return 0;
    uint64_t low_dim, group_stride, heads_bytes, low_bytes, tmp_heads_bytes, tmp_low_bytes;
    if (!checked_u64_product(n_groups, rank, low_dim) || low_dim > UINT32_MAX ||
        !checked_u64_product(n_groups, group_dim, group_stride) || group_stride > UINT32_MAX ||
        !checked_f32_bytes(n_tokens, n_groups, group_dim, heads_bytes) ||
        !checked_f32_bytes(n_tokens, low_dim, 1, low_bytes) ||
        !checked_f32_bytes(n_tokens, group_dim, 1, tmp_heads_bytes) ||
        !checked_f32_bytes(n_tokens, rank, 1, tmp_low_bytes) ||
        heads_bytes > heads->bytes || low_bytes > low->bytes) return 0;
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    if (blocks_a == 0 || low_dim > UINT64_MAX / (blocks_a * 34u)) return 0;
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t out_a_bytes = low_dim * row_a_bytes;
    if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset) return 0;

    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;

    /* Decode has one head row per group and the graph stores those rows
     * contiguously: group g begins at g*group_dim floats and the destination
     * low vector begins at g*rank.  Preserve the batched/prefill path below,
     * where groups are strided across tokens, but use aligned tensor views for
     * the single-token decode shape.  The projection sees exactly the same
     * bytes and accumulation order. */
    if (n_tokens == 1) {
        /* All eight group-A GEMVs consume rows of the same activation.  The
         * ordinary Q8 path quantizes each row on every call, creating eight
         * Q8 temporaries and eight quantize dispatches.  Quantize the complete
         * contiguous group-row span once, then run the grouped prequant
         * appliance over the same eight weight-row ranges.  Each output row
         * keeps its established shader reduction order. */
        const uint64_t q_row_bytes = blocks_a * 36u;
        const char *q8_mode = getenv("DS4_VULKAN_Q8_MODE");
        const bool shared_q8_eligible =
            !(q8_mode && strcmp(q8_mode, "exact") == 0) &&
            n_groups <= 65535u && blocks_a <= 256u &&
            n_groups <= UINT64_MAX / blocks_a &&
            n_groups * blocks_a <= UINT64_MAX / 36u &&
            n_groups * blocks_a * 36u <= UINT32_MAX &&
            g_vk.caps.max_compute_work_group_size[0] >= 256u &&
            g_vk.caps.max_compute_work_group_invocations >= 256u;
        bool grouped_ready = shared_q8_eligible;
        if (grouped_ready) {
            /* Complete this preflight before recording quantization.  A
             * missing bundled shader or an unavailable full-span aligned
             * artifact must select the original eight-call path, rather than
             * leaving a partially recorded grouped graph with no fallback. */
            const uint64_t grouped_rows = (uint64_t)n_groups * rank;
            auto grouped_shader = g_vk.shader_map.find("matmul_q8_0_group_bfe");
            auto grouped_rows_shader = g_vk.shader_map.find("matmul_q8_0_group_rows_bfe");
            decltype(g_vk.aligned_cache)::mapped_type *grouped_aligned = nullptr;
            grouped_ready = (grouped_shader != g_vk.shader_map.end() ||
                             grouped_rows_shader != g_vk.shader_map.end()) &&
                grouped_rows != 0 &&
                grouped_rows <= UINT64_MAX / row_a_bytes &&
                out_a_offset <= model_size &&
                grouped_rows * row_a_bytes <= model_size - out_a_offset &&
                rank <= g_vk.caps.max_compute_work_group_count[0] &&
                n_groups <= g_vk.caps.max_compute_work_group_count[1] &&
                ensure_aligned_weight(model_map, model_size, out_a_offset,
                                      group_dim, grouped_rows, grouped_aligned) &&
                grouped_aligned != nullptr;
        }
        if (grouped_ready) {
            const uint64_t q_bytes = (uint64_t)n_groups * q_row_bytes;
            ds4_gpu_tensor *q = ds4_gpu_tensor_alloc(q_bytes);
            if (q) {
                int ok = ds4_gpu_quantize_q8_0_tensor(q, heads,
                                                       group_dim, n_groups);
                if (ok) ok = ds4_gpu_matmul_q8_0_group_tensor(
                    low, model_map, model_size, out_a_offset,
                    group_dim, rank, n_groups, q);

                if (ctx.attention_output_batch) {
                    free_or_defer_attention_tensor(ctx, q);
                    if (resume_recording && !ctx.recording && !begin_cmd()) ok = 0;
                    return ok;
                }
                if (ok) {
                    /* Keep q alive until every recorded GEMV has consumed it. */
                    ok = submit_and_wait();
                } else if (ctx.recording && ctx.command_count != 0) {
                    submit_and_wait();
                }
                ds4_gpu_tensor_free(q);
                if (resume_recording && !ctx.recording && !begin_cmd()) ok = 0;
                return ok;
            }
        }

        /* Capability, allocation, or stale-shader fallback: retain the
         * original one-row path rather than changing the production contract. */
        int ok = 1;
        for (uint32_t g = 0; g < n_groups && ok; g++) {
            ds4_gpu_tensor heads_view = *heads;
            heads_view.ptr = (char *)heads->ptr +
                (uint64_t)g * group_dim * sizeof(float);
            heads_view.bytes = heads_bytes -
                (uint64_t)g * group_dim * sizeof(float);
            ds4_gpu_tensor low_view = *low;
            low_view.ptr = (char *)low->ptr +
                (uint64_t)g * rank * sizeof(float);
            low_view.bytes = low_bytes - (uint64_t)g * rank * sizeof(float);
            const uint64_t a_offset = out_a_offset +
                (uint64_t)g * rank * row_a_bytes;
            ok = ds4_gpu_matmul_q8_0_tensor(&low_view, model_map, model_size,
                                            a_offset, group_dim, rank,
                                            &heads_view, 1);
        }
        if (resume_recording && !ctx.recording && !begin_cmd()) ok = 0;
        return ok;
    }

    ds4_gpu_tensor *group_heads = ds4_gpu_tensor_alloc(tmp_heads_bytes);
    ds4_gpu_tensor *group_low = ds4_gpu_tensor_alloc(tmp_low_bytes);
    if (!group_heads || !group_low) {
        ds4_gpu_tensor_free(group_low); ds4_gpu_tensor_free(group_heads); return 0;
    }
    int ok = 1;
    for (uint32_t g = 0; g < n_groups && ok; g++) {
        ds4_gpu_tensor heads_view = *heads;
        heads_view.ptr = (char *)heads->ptr + (uint64_t)g * group_dim * sizeof(float);
        heads_view.bytes = heads_bytes - (uint64_t)g * group_dim * sizeof(float);
        ds4_gpu_tensor low_view = *low;
        low_view.ptr = (char *)low->ptr + (uint64_t)g * rank * sizeof(float);
        low_view.bytes = low_bytes - (uint64_t)g * rank * sizeof(float);
        const uint64_t a_offset = out_a_offset + (uint64_t)g * rank * row_a_bytes;
        ok = ds4_vk_group_copy(&heads_view, group_heads, (uint32_t)group_dim,
                       (uint32_t)group_stride, (uint32_t)group_dim,
                               n_tokens);
        if (ok) ok = ds4_gpu_matmul_q8_0_tensor(group_low, model_map, model_size,
                                                a_offset, group_dim, rank,
                                                group_heads, n_tokens);
        if (ok) ok = ds4_vk_group_copy(group_low, &low_view, (uint32_t)rank,
                                       (uint32_t)rank, (uint32_t)low_dim, n_tokens);
    }
    free_or_defer_attention_tensor(ctx, group_low);
    free_or_defer_attention_tensor(ctx, group_heads);
    if (resume_recording && !ctx.recording && !begin_cmd()) ok = 0;
    return ok;
}

int ds4_gpu_attention_output_low_q8_tensor(ds4_gpu_tensor *low,
    const void *model_map, uint64_t model_size, uint64_t out_a_offset,
    uint64_t group_dim, uint64_t rank, uint32_t n_groups,
    const ds4_gpu_tensor *heads)
{
    return ds4_vk_attention_output_low_gpu(low, model_map, model_size,
                                           out_a_offset, group_dim, rank,
                                           n_groups, heads, 1);
}

int ds4_gpu_attention_output_q8_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
    ds4_gpu_tensor *gt, ds4_gpu_tensor *lt, const void *mm, uint64_t ms,
    uint64_t oa_off, uint64_t ob_off, uint64_t gd, uint64_t rank,
    uint32_t ng, uint64_t od, const ds4_gpu_tensor *heads, uint32_t nt)
{
    (void)gt; (void)lt;
    if (!out || !low || !heads || !mm || gd == 0 || rank == 0 ||
        ng == 0 || od == 0 || nt == 0) return 0;
    uint64_t low_dim;
    if (!checked_u64_product(ng, rank, low_dim)) return 0;
    auto &ctx = get_cmd_ctx();
    const char *batch_env = getenv("DS4_VULKAN_BATCH_ATTENTION_OUTPUT");
    const bool batch = nt == 1 &&
        (!batch_env || !batch_env[0] || strcmp(batch_env, "0") != 0);
    if (!batch) {
        if (!ds4_vk_attention_output_low_gpu(low, mm, ms, oa_off, gd, rank,
                                             ng, heads, nt)) return 0;
        return ds4_gpu_matmul_q8_0_tensor(out, mm, ms, ob_off, low_dim, od, low, nt);
    }

    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    ctx.attention_output_descriptors.clear();
    ctx.attention_output_tensors.clear();
    ctx.attention_output_descriptors.reserve(34);
    ctx.attention_output_tensors.reserve(16);
    ctx.attention_output_batch = true;

    int ok = ds4_vk_attention_output_low_gpu(low, mm, ms, oa_off, gd, rank,
                                              ng, heads, nt);
    if (ok)
        ok = ds4_gpu_matmul_q8_0_tensor(out, mm, ms, ob_off,
                                       low_dim, od, low, nt);
    ctx.attention_output_batch = false;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) ok = 0;
    for (VkDescriptorSet set : ctx.attention_output_descriptors)
        if (!release_simple_descriptors(set)) ok = 0;
    for (ds4_gpu_tensor *tensor : ctx.attention_output_tensors)
        ds4_gpu_tensor_free(tensor);
    ctx.attention_output_descriptors.clear();
    ctx.attention_output_tensors.clear();
    if (resume_recording && !ctx.recording && !begin_cmd()) ok = 0;
    return ok;
}

/* The Vulkan backend declines the F16-output shortcut so the caller can use
 * the canonical F32 projection path instead. */
int ds4_gpu_attention_output_q8_batch_f16_tensor(
        ds4_gpu_tensor       *out_h,
        ds4_gpu_tensor       *low,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                out_a_offset,
        uint64_t                out_b_offset,
        uint64_t                group_dim,
        uint64_t                rank,
        uint32_t                n_groups,
        uint64_t                out_dim,
        const ds4_gpu_tensor *heads,
        uint32_t                n_tokens)
{
    (void)out_h; (void)low; (void)model_map; (void)model_size;
    (void)out_a_offset; (void)out_b_offset; (void)group_dim; (void)rank;
    (void)n_groups; (void)out_dim; (void)heads; (void)n_tokens;
    DS4_VK_TRACE_KERNEL("attention_output_q8_batch_f16_unsupported");
    return 0;
}

static thread_local bool g_indexed_wave64_used = false;
static thread_local bool g_indexed_wave64_inv_rope_used = false;

/* BC-250 exposes the GFX1013/Radeon 890M identity through RADV.  Keep the
 * tuned indexed path opt-in on every other device, while allowing explicit
 * 1/0 overrides for bring-up and rollback. */
static bool indexed_wave64_bc250_default(void) {
    if (g_vk.caps.subgroup_size != 64u || !g_vk.caps.has_subgroup_shuffle)
        return false;
    const char *name = ds4_vulkan_gpu_name;
    return name != nullptr &&
        (strstr(name, "890M") != nullptr ||
         strstr(name, "BC-250") != nullptr ||
         strstr(name, "BC250") != nullptr);
}

static bool indexed_wave64_enabled(void) {
    const char *env = getenv("DS4_VULKAN_ATTN_INDEXED_WAVE64");
    return env != nullptr ? strcmp(env, "0") != 0
                          : indexed_wave64_bc250_default();
}

static bool indexed_wave64_inv_rope_enabled(void) {
    const char *env = getenv("DS4_VULKAN_ATTN_INDEXED_WAVE64_INV_ROPE");
    return env != nullptr ? strcmp(env, "0") != 0
                          : indexed_wave64_bc250_default();
}

extern "C" int ds4_gpu_attention_indexed_wave64_used(void) {
    return g_indexed_wave64_used ? 1 : 0;
}

extern "C" int ds4_gpu_attention_indexed_wave64_inv_rope_available(void) {
    return indexed_wave64_enabled() && indexed_wave64_inv_rope_enabled() &&
        g_vk.shader_map.find("attention_indexed_online_wave64") !=
            g_vk.shader_map.end();
}

extern "C" int ds4_gpu_attention_indexed_wave64_inv_rope_used(void) {
    return g_indexed_wave64_inv_rope_used ? 1 : 0;
}

static int dispatch_attention_mixed_online(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv, const ds4_gpu_tensor *comp_kv,
        uint32_t comp_kv_f16, const ds4_gpu_tensor *topk,
        const ds4_gpu_tensor *comp_mask, uint32_t use_mask,
        uint32_t n_tokens, uint32_t pos0, uint32_t q_row0, uint32_t n_q,
        uint32_t n_raw, uint32_t raw_cap, uint32_t raw_start,
        uint32_t n_comp, uint32_t top_k, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim, uint32_t mode) {
    if (mode == 1u) {
        g_indexed_wave64_used = false;
        g_indexed_wave64_inv_rope_used = false;
    }
    uint64_t head_count, head_bytes, q_bytes, raw_bytes, comp_values;
    uint64_t comp_bytes, mask_bytes, topk_values, topk_bytes, sink_bytes;
    const uint64_t position_end = (uint64_t)pos0 + q_row0 + n_q;
    if (!heads || !q || !raw_kv || !model_map || n_q == 0 || n_tokens == 0 ||
        n_head == 0 || head_dim == 0 || n_q > n_tokens ||
        q_row0 > n_tokens - n_q || n_raw > raw_cap ||
        (n_raw != 0 && raw_start >= raw_cap) || n_comp > 4096u ||
        top_k > 512u || (mode == 1u && top_k == 0u) ||
        (n_comp != 0 && !comp_kv) || (mode == 1u && !topk) ||
        (use_mask != 0u && !comp_mask) ||
        position_end > UINT32_MAX || pos0 > UINT32_MAX - n_tokens ||
        (uint64_t)n_raw > (uint64_t)pos0 + n_tokens ||
        (n_raw != 0u && raw_start > UINT32_MAX - n_raw) ||
        !checked_u64_product(n_q, n_head, head_count) || head_count > UINT32_MAX ||
        !shader_f32_domain(n_q, n_head, head_dim) ||
        !shader_f32_domain(n_tokens, n_head, head_dim) ||
        !shader_f32_domain(raw_cap, 1, head_dim) ||
        !shader_f32_domain(n_comp, 1, head_dim) ||
        (mode == 1u && !shader_f32_domain(n_tokens, top_k, 1)) ||
        !checked_f32_bytes(n_q, n_head, head_dim, head_bytes) ||
        !checked_f32_bytes(n_q, n_head, head_dim, q_bytes) ||
        !checked_f32_bytes(raw_cap, 1, head_dim, raw_bytes) ||
        !checked_u64_product(n_comp, head_dim, comp_values) ||
        !checked_u64_product(n_comp, sizeof(float), mask_bytes) ||
        !checked_u64_product(n_tokens, top_k, topk_values) ||
        !checked_u64_product(topk_values, sizeof(uint32_t), topk_bytes) ||
        !checked_u64_product(n_head, sizeof(float), sink_bytes))
        return 0;
    if (comp_kv_f16) {
        if (comp_values > UINT64_MAX - 1u ||
            !checked_u64_product((comp_values + 1u) / 2u, sizeof(uint32_t), comp_bytes))
            return 0;
    } else if (!checked_u64_product(comp_values, sizeof(float), comp_bytes)) {
        return 0;
    }
    uint64_t max_visible = 0;
    if (ratio != 0) {
        max_visible = std::min<uint64_t>((position_end - 1u) / ratio, n_comp);
    }
    const uint64_t visible_limit = (mode == 0u && use_mask == 0u) ? 512u : 4096u;
    if (max_visible > visible_limit || raw_bytes < 4u) return 0;
    if (heads->bytes < head_bytes || q->bytes < q_bytes ||
        raw_kv->bytes < raw_bytes ||
        (n_comp && comp_kv->bytes < comp_bytes) ||
        (mode == 1u && topk->bytes < topk_bytes) ||
        (use_mask && comp_mask->bytes < mask_bytes) ||
        sinks_offset > model_size || sink_bytes > model_size - sinks_offset)
        return 0;
    VkBuffer obuf, qbuf, rbuf, cbuf, tbuf, mbuf, sbuf;
    VkDeviceSize ooff, qoff, roff, coff, toff, moff, soff, ssize;
    if (!find_tensor_buffer(heads, obuf, ooff) || !find_tensor_buffer(q, qbuf, qoff) ||
        !find_tensor_buffer(raw_kv, rbuf, roff) ||
        !find_model_buffer(sinks_offset, sink_bytes, sbuf, soff, ssize)) return 0;
    if (n_comp && !find_tensor_buffer(comp_kv, cbuf, coff)) return 0;
    if (!n_comp) { cbuf = rbuf; coff = roff; }
    if (mode == 1u && !find_tensor_buffer(topk, tbuf, toff)) return 0;
    if (mode != 1u) { tbuf = rbuf; toff = roff; }
    if (use_mask && !find_tensor_buffer(comp_mask, mbuf, moff)) return 0;
    if (!use_mask) { mbuf = rbuf; moff = roff; }
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((ooff | qoff | roff | coff | toff | moff | soff) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    constexpr uint32_t max_query_tokens_per_dispatch = 256u;
    const uint64_t row_bytes = (uint64_t)n_head * head_dim * sizeof(float);
    const uint64_t topk_row_bytes = (uint64_t)top_k * sizeof(uint32_t);
    for (uint32_t local_row0 = 0; local_row0 < n_q; ) {
        const uint32_t tile_rows = std::min(max_query_tokens_per_dispatch,
                                            n_q - local_row0);
        if (!ctx.recording && !begin_cmd()) return 0;
        if ((uint64_t)local_row0 > UINT64_MAX / row_bytes ||
            (mode == 1u && (uint64_t)local_row0 > UINT64_MAX / topk_row_bytes))
            return fail_simple_dispatch(ctx);
        const uint64_t row_delta = (uint64_t)local_row0 * row_bytes;
        const uint64_t topk_delta = mode == 1u
            ? (uint64_t)local_row0 * topk_row_bytes : 0u;
        if (row_delta > UINT64_MAX - ooff || row_delta > UINT64_MAX - qoff ||
            topk_delta > UINT64_MAX - toff)
            return fail_simple_dispatch(ctx);
        const VkDeviceSize tile_ooff = ooff + row_delta;
        const VkDeviceSize tile_qoff = qoff + row_delta;
        const VkDeviceSize tile_toff = toff + topk_delta;
        if (align && ((tile_ooff | tile_qoff | tile_toff) % align) != 0)
            return fail_simple_dispatch(ctx);
        VkDescriptorBufferInfo bufs[8] = {
            {obuf, tile_ooff, (VkDeviceSize)tile_rows * row_bytes},
            {qbuf, tile_qoff, (VkDeviceSize)tile_rows * row_bytes},
            {rbuf, roff, (VkDeviceSize)raw_kv->bytes},
            {cbuf, coff, n_comp ? (VkDeviceSize)comp_kv->bytes : 4},
            {tbuf, tile_toff, mode == 1u
                ? (VkDeviceSize)tile_rows * topk_row_bytes : 4},
            {mbuf, moff, use_mask ? (VkDeviceSize)comp_mask->bytes : 4},
            {sbuf, soff, ssize}, {rbuf, roff, 4}
        };
        struct { uint32_t n_tokens, pos0, q_row0, n_q, n_raw, raw_cap, raw_start, n_comp,
                 top_k, window, ratio, n_head, head_dim, comp_f16, use_mask, mode; } pc = {
            n_tokens, pos0, q_row0 + local_row0, tile_rows, n_raw, raw_cap,
            raw_start, n_comp, top_k, window, ratio, n_head, head_dim,
            comp_kv_f16, use_mask, mode};
        const uint64_t tile_head_count = (uint64_t)tile_rows * n_head;
        const bool more_tiles = local_row0 + tile_rows < n_q;
        /* Indexed 128K decode is the only path admitted to the register
         * accumulator candidate.  It preserves the canonical online order,
         * but requires one full wave and exactly two head values per lane. */
        const bool use_indexed_wave64 = mode == 1u && head_dim == 128u &&
            indexed_wave64_enabled() &&
            g_vk.shader_map.find("attention_indexed_online_wave64") !=
                g_vk.shader_map.end();
        const bool use_indexed_wave64_inv_rope = use_indexed_wave64 &&
            n_tokens == 1u && n_q == 1u &&
            g_decode_attn_rope_fuse.armed &&
            g_decode_attn_rope_fuse.n_rot == 64u &&
            g_decode_attn_rope_fuse.inverse &&
            indexed_wave64_inv_rope_enabled();
        struct RopePC {
            uint32_t n_tokens, pos0, q_row0, n_q, n_raw, raw_cap, raw_start, n_comp,
                top_k, window, ratio, n_head, head_dim, comp_f16, use_mask, mode;
            uint32_t rope_enable, n_rot, rope_pos0, n_ctx_orig;
            int32_t inverse;
            float freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow;
        } rope_pc = {
            pc.n_tokens, pc.pos0, pc.q_row0, pc.n_q, pc.n_raw, pc.raw_cap,
            pc.raw_start, pc.n_comp, pc.top_k, pc.window, pc.ratio, pc.n_head,
            pc.head_dim, pc.comp_f16, pc.use_mask, pc.mode,
            1u, g_decode_attn_rope_fuse.n_rot,
            g_decode_attn_rope_fuse.pos0 + local_row0,
            g_decode_attn_rope_fuse.n_ctx_orig,
            g_decode_attn_rope_fuse.inverse ? 1 : 0,
            g_decode_attn_rope_fuse.freq_base, g_decode_attn_rope_fuse.freq_scale,
            g_decode_attn_rope_fuse.ext_factor, g_decode_attn_rope_fuse.attn_factor,
            g_decode_attn_rope_fuse.beta_fast, g_decode_attn_rope_fuse.beta_slow};
        /* The indexed Wave64 shader always has the extended push-constant
         * layout, even when inverse RoPE is not fused.  Do not leave the
         * trailing bytes stale: push constants persist across dispatches in a
         * command buffer, so a prior fused dispatch could otherwise make a
         * later ordinary indexed dispatch rotate its output accidentally. */
        if (!use_indexed_wave64_inv_rope) {
            rope_pc.rope_enable = 0u;
            rope_pc.n_rot = 0u;
            rope_pc.rope_pos0 = 0u;
            rope_pc.n_ctx_orig = 0u;
            rope_pc.inverse = 0;
            rope_pc.freq_base = 0.0f;
            rope_pc.freq_scale = 1.0f;
            rope_pc.ext_factor = 0.0f;
            rope_pc.attn_factor = 1.0f;
            rope_pc.beta_fast = 0.0f;
            rope_pc.beta_slow = 0.0f;
        }
        const char *shader_name = use_indexed_wave64
            ? "attention_indexed_online_wave64" : "attention_mixed_online";
        if (use_indexed_wave64) g_indexed_wave64_used = true;
        DS4_VK_TRACE_KERNEL(shader_name);
        if (tile_head_count > g_vk.caps.max_compute_work_group_count[0] ||
            !record_simple_shader(
                shader_name,
                use_indexed_wave64 ? (const void *)&rope_pc : (const void *)&pc,
                use_indexed_wave64 ? sizeof(rope_pc) : sizeof(pc),
                bufs, 8, (uint32_t)tile_head_count, 1, 1,
                more_tiles || resume_recording)) return 0;
        if (use_indexed_wave64_inv_rope) {
            g_indexed_wave64_inv_rope_used = true;
            g_decode_attn_rope_fuse.armed = false;
            g_decode_attn_rope_fuse.used = true;
        }
        local_row0 += tile_rows;
    }
    return 1;
}

int ds4_gpu_attention_decode_mixed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16,
        const ds4_gpu_tensor *comp_mask, uint32_t use_comp_mask,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, uint32_t n_comp, uint32_t window, uint32_t ratio,
        uint32_t n_head, uint32_t head_dim) {
    return dispatch_attention_mixed_online(heads, model_map, model_size, sinks_offset,
        q, raw_kv, comp_kv, comp_kv_f16, nullptr, comp_mask, use_comp_mask,
        n_tokens, pos0, 0, n_tokens, n_raw, raw_cap, raw_start, n_comp, 0,
        window, ratio, n_head, head_dim, 2);
}

int ds4_gpu_attention_indexed_mixed_batch_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16, const ds4_gpu_tensor *topk,
        uint32_t n_tokens, uint32_t pos0, uint32_t n_raw, uint32_t raw_cap,
        uint32_t raw_start, uint32_t n_comp, uint32_t top_k, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    return dispatch_attention_mixed_online(heads, model_map, model_size, sinks_offset,
        q, raw_kv, comp_kv, comp_kv_f16, topk, nullptr, 0, n_tokens, pos0, 0,
        n_tokens, n_raw, raw_cap, raw_start, n_comp, top_k, window, ratio,
        n_head, head_dim, 1);
}

/* ---- ds4_gpu_attention_prefill_static_mixed_heads_tensor ----
 *
 * Prefill attention over the mixed key set: the raw batch KV rows of the
 * current chunk (MLA-style: one row per token, K and V share the row, as in
 * the verified ds4_gpu_attention_prefill_raw_heads_tensor) plus the
 * compressed MLA rows of the earlier prefix, with a per-head sink prior and
 * a causal mask (token t only sees keys at positions <= t).
 *
 * Replicates the ROCm attention_static_mixed_heads8_online_kernel and the
 * verified attention_prefill_raw_heads_tensor / attention_decode_heads_tensor
 * math exactly:
 *   raw_count = (window != 0 && t+1 > window) ? window : t+1
 *   raw_start = t+1 - raw_count
 *   comp_count = (n_comp != 0 && ratio != 0) ? min((t+1)/ratio, n_comp) : 0
 *   scale = 1/sqrtf(head_dim)
 *   per head h: max starts at sinks[h]; raw score  = dot(qh, raw[raw_start+r]) * scale
 *               comp score = dot(qh, comp[c]) * scale (comp rows f32 or f16)
 *   heads[t][h] = sum(exp(score - max) * v) / (exp(sinks[h]-max) + sum exp(...))
 */
int ds4_gpu_attention_prefill_static_mixed_heads_tensor(
        ds4_gpu_tensor       *heads,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                sinks_offset,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv,
        uint32_t                comp_kv_f16,
        uint32_t                n_tokens,
        uint32_t                n_comp,
        uint32_t                window,
        uint32_t                ratio,
        uint32_t                n_head,
        uint32_t                head_dim)
{
    return dispatch_attention_mixed_online(heads, model_map, model_size, sinks_offset,
        q, raw_kv, comp_kv, comp_kv_f16, nullptr, nullptr, 0,
        n_tokens, 0, 0, n_tokens, n_tokens, n_tokens, 0, n_comp, 0,
        window, ratio, n_head, head_dim, 0);
}

int ds4_gpu_attention_prefill_static_mixed_heads_range_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16, uint32_t q_row0,
        uint32_t n_q, uint32_t n_tokens, uint32_t n_comp, uint32_t window,
        uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    return dispatch_attention_mixed_online(heads, model_map, model_size, sinks_offset,
        q, raw_kv, comp_kv, comp_kv_f16, nullptr, nullptr, 0, n_tokens, 0,
        q_row0, n_q, n_tokens, n_tokens, 0, n_comp, 0, window, ratio,
        n_head, head_dim, 0);
}

int ds4_gpu_attention_prefill_masked_mixed_heads_tensor(
        ds4_gpu_tensor *heads, const void *model_map, uint64_t model_size,
        uint64_t sinks_offset, const ds4_gpu_tensor *q, const ds4_gpu_tensor *raw_kv,
        const ds4_gpu_tensor *comp_kv, uint32_t comp_kv_f16,
        const ds4_gpu_tensor *comp_mask, uint32_t n_tokens, uint32_t n_comp,
        uint32_t window, uint32_t ratio, uint32_t n_head, uint32_t head_dim) {
    return dispatch_attention_mixed_online(heads, model_map, model_size, sinks_offset,
        q, raw_kv, comp_kv, comp_kv_f16, nullptr, comp_mask, 1, n_tokens, 0,
        0, n_tokens, n_tokens, n_tokens, 0, n_comp, 0, window, ratio,
        n_head, head_dim, 0);
}

/* ---- Hyper-Connection helpers ----
 *
 * HC split, weighted reduction, and fused norm use Vulkan compute below.
 * The scalar sigmoid remains here for the output-head helper.
 */

static int dispatch_hc_weighted_sum(ds4_gpu_tensor *out, const ds4_gpu_tensor *rhc,
    const ds4_gpu_tensor *weights, uint32_t n_embd, uint32_t n_hc, uint32_t rows) {
    if (!out || !rhc || !weights || n_embd == 0 || n_hc == 0 || rows == 0 ||
        rows > 65535u) return 0;
    auto si = g_vk.shader_map.find("hc_weighted_sum");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer obuf, rbuf, wbuf; VkDeviceSize ooff, roff, woff;
    if (!find_tensor_buffer(out, obuf, ooff) ||
        !find_tensor_buffer(rhc, rbuf, roff) ||
        !find_tensor_buffer(weights, wbuf, woff)) return 0;
    const VkDeviceSize out_bytes = (VkDeviceSize)rows * n_embd * sizeof(float);
    const VkDeviceSize residual_bytes = (VkDeviceSize)rows * n_hc * n_embd * sizeof(float);
    const VkDeviceSize weight_bytes = (VkDeviceSize)rows * n_hc * sizeof(float);
    VkDescriptorBufferInfo buffers[3] = {
        {obuf, ooff, out_bytes}, {rbuf, roff, residual_bytes}, {wbuf, woff, weight_bytes}
    };
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, 3, set))
        return fail_simple_dispatch(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    struct { uint32_t n_embd, n_hc, n_rows, reserved; } push = {n_embd, n_hc, rows, 0};
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    timeline_dispatch(ctx, "hc_weighted_sum", buffers, 3,
                      (n_embd + 255u) / 256u, rows, 1);
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_simple_descriptors(set)) ok = 0;
    return ok;
}

static int dispatch_hc_expand(ds4_gpu_tensor *out, const ds4_gpu_tensor *block,
    const ds4_gpu_tensor *add, const ds4_gpu_tensor *residual,
    const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
    uint32_t n_embd, uint32_t n_hc, uint32_t rows, bool split_layout,
    bool block_is_half, bool add_is_half) {
    if (!out || !block || !residual || !post || !comb ||
        n_embd == 0 || n_hc == 0 || rows == 0 ||
        (block_is_half && (n_embd & 1u) != 0) ||
        (add_is_half && (n_embd & 1u) != 0) ||
        n_hc > UINT32_MAX / n_hc ||
        2ull * n_hc + (uint64_t)n_hc * n_hc > UINT32_MAX ||
        rows > 65535u / n_hc) return 0;
    auto si = g_vk.shader_map.find("hc_expand");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer obuf, bbuf, abuf, rbuf, pbuf, cbuf;
    VkDeviceSize ooff, boff, aoff, roff, poff, coff;
    if (!find_tensor_buffer(out, obuf, ooff) ||
        !find_tensor_buffer(block, bbuf, boff) ||
        !find_tensor_buffer(residual, rbuf, roff) ||
        !find_tensor_buffer(post, pbuf, poff) ||
        !find_tensor_buffer(comb, cbuf, coff)) return 0;
    if (add) {
        if (!find_tensor_buffer(add, abuf, aoff)) return 0;
    } else {
        abuf = bbuf; aoff = boff;
    }
    const uint64_t block_elem = block_is_half ? sizeof(uint16_t) : sizeof(float);
    const uint64_t add_elem = add_is_half ? sizeof(uint16_t) : sizeof(float);
    const uint32_t post_stride = split_layout ?
        (uint32_t)(2ull * n_hc + (uint64_t)n_hc * n_hc) : n_hc;
    const uint32_t comb_stride = split_layout ? post_stride : n_hc * n_hc;
    VkDescriptorBufferInfo buffers[6] = {
        {obuf, ooff, (VkDeviceSize)rows * n_hc * n_embd * sizeof(float)},
        {bbuf, boff, (VkDeviceSize)rows * n_embd * block_elem},
        {abuf, aoff, (VkDeviceSize)rows * n_embd * add_elem},
        {rbuf, roff, (VkDeviceSize)rows * n_hc * n_embd * sizeof(float)},
        {pbuf, poff, (VkDeviceSize)rows * post_stride * sizeof(float)},
        {cbuf, coff, (VkDeviceSize)rows * comb_stride * sizeof(float)},
    };
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, 6, set))
        return fail_simple_dispatch(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    struct {
        uint32_t n_embd, n_hc, n_rows, post_stride, comb_stride;
        uint32_t has_add, has_add2, split_layout, block_is_half;
    } push = {n_embd, n_hc, rows, post_stride, comb_stride,
              add ? 1u : 0u, add && add_is_half ? 1u : 0u,
              split_layout ? 1u : 0u, block_is_half ? 1u : 0u};
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    timeline_dispatch(ctx, "hc_expand", buffers, 6,
                      (n_embd + 255u) / 256u, rows * n_hc, 1);
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_simple_descriptors(set)) ok = 0;
    return ok;
}

int ds4_gpu_hc_weighted_sum_split_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *rhc,
    const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc)
{
    if (!out || !rhc || !split || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t hc_row = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t weight_row = (uint64_t)n_hc * sizeof(float);
    uint64_t rows = std::min(out->bytes / ((uint64_t)n_embd * sizeof(float)),
                             rhc->bytes / hc_row);
    rows = std::min(rows, split->bytes / weight_row);
    return rows > UINT32_MAX ? 0 : dispatch_hc_weighted_sum(out, rhc, split,
                                                             n_embd, n_hc, (uint32_t)rows);
}

static int hc_cached_weight(uint64_t offset, uint64_t bytes,
                            VkBuffer &buffer, VkDeviceSize &buffer_offset) {
    if (!ensure_weight(offset, bytes)) return 0;
    for (auto it = g_vk.weight_cache.begin(); it != g_vk.weight_cache.end(); ++it) {
        if (offset >= it->first && offset - it->first <= it->second.size &&
            bytes <= it->second.size - (offset - it->first)) {
            const VkDeviceSize relative = (VkDeviceSize)(offset - it->first);
            const VkDeviceSize alignment =
                (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
            if (alignment != 0 && relative % alignment != 0) return 0;
            it->second.last_used = ++g_vk.lru_counter;
            mark_weight_generation(it->second);
            buffer = it->second.buffer;
            buffer_offset = relative;
            return 1;
        }
    }
    return 0;
}

static int dispatch_hc_split_weighted_sum(ds4_gpu_tensor *out,
    ds4_gpu_tensor *norm_out, ds4_gpu_tensor *split,
    const ds4_gpu_tensor *mix, const ds4_gpu_tensor *rhc,
    const void *model_map, uint64_t model_size, uint64_t scale_offset,
    uint64_t base_offset, uint64_t norm_weight_offset, uint32_t n_embd,
    uint32_t n_hc, uint32_t sinkhorn_iters, float eps, float norm_eps,
    bool do_sum, bool do_norm) {
    if (!out || !split || !mix || !model_map || n_hc == 0 || n_hc > 16 ||
        n_embd == 0 || sinkhorn_iters == 0 || (do_sum && !rhc) ||
        (do_norm && (!norm_out || norm_eps < 0.0f))) return 0;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t split_bytes = mix_hc * sizeof(float);
    const uint64_t out_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t rhc_row_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    uint64_t rows = std::min(split->bytes / split_bytes, mix->bytes / split_bytes);
    if (do_sum) {
        rows = std::min(rows, out->bytes / out_row_bytes);
        rows = std::min(rows, rhc->bytes / rhc_row_bytes);
    } else {
        rows = std::min(rows, out->bytes / split_bytes);
    }
    if (do_norm) rows = std::min(rows, norm_out->bytes / out_row_bytes);
    if (rows == 0 || rows > 65535u) return 0;
    if (scale_offset > model_size || 3ull * sizeof(float) > model_size - scale_offset ||
        base_offset > model_size || split_bytes > model_size - base_offset) return 0;
    if (do_norm && (norm_weight_offset > model_size ||
                    out_row_bytes > model_size - norm_weight_offset)) return 0;
    if (g_vk.model_map != model_map || g_vk.model_size != model_size)
        set_model_map_identity(model_map, model_size);

    VkBuffer obuf, nbuf, sbuf, mbuf, rbuf, scale_buf, base_buf, norm_buf;
    VkDeviceSize ooff, noff, soff, moff, roff, scale_off, base_off, norm_off;
    if (!find_tensor_buffer(out, obuf, ooff) ||
        !find_tensor_buffer(split, sbuf, soff) ||
        !find_tensor_buffer(mix, mbuf, moff)) return 0;
    if (do_sum) {
        if (!find_tensor_buffer(rhc, rbuf, roff)) return 0;
    } else {
        rbuf = mbuf; roff = moff;
    }
    if (do_norm) {
        if (!find_tensor_buffer(norm_out, nbuf, noff)) return 0;
    } else {
        nbuf = obuf; noff = ooff;
    }
    if (!hc_cached_weight(scale_offset, 3ull * sizeof(float), scale_buf, scale_off) ||
        !hc_cached_weight(base_offset, split_bytes, base_buf, base_off)) return 0;
    if (do_norm) {
        if (!hc_cached_weight(norm_weight_offset, out_row_bytes, norm_buf, norm_off)) return 0;
    } else {
        norm_buf = scale_buf; norm_off = scale_off;
    }

    auto si = g_vk.shader_map.find("hc_split_weighted_sum");
    if (si == g_vk.shader_map.end()) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorBufferInfo buffers[8] = {
        {obuf, ooff, do_sum ? (VkDeviceSize)rows * out_row_bytes : (VkDeviceSize)split_bytes},
        {nbuf, noff, do_norm ? (VkDeviceSize)rows * out_row_bytes : (VkDeviceSize)split_bytes},
        {sbuf, soff, (VkDeviceSize)rows * split_bytes},
        {mbuf, moff, (VkDeviceSize)rows * split_bytes},
        {rbuf, roff, do_sum ? (VkDeviceSize)rows * rhc_row_bytes : (VkDeviceSize)split_bytes},
        {scale_buf, scale_off, 3 * sizeof(float)},
        {base_buf, base_off, split_bytes},
        {norm_buf, norm_off, do_norm ? (VkDeviceSize)out_row_bytes : 3 * sizeof(float)},
    };
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, 8, set))
        return fail_simple_dispatch(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    struct { uint32_t n_embd, n_hc, n_rows, sinkhorn_iters;
             float eps, norm_eps; uint32_t has_sum, has_norm; } push = {
        n_embd, n_hc, (uint32_t)rows, sinkhorn_iters, eps, norm_eps,
        do_sum ? 1u : 0u, do_norm ? 1u : 0u};
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    timeline_dispatch(ctx, "hc_split_weighted_sum", buffers, 8,
                      1, (uint32_t)rows, 1);
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_simple_descriptors(set)) ok = 0;
    return ok;
}

int ds4_gpu_hc_split_sinkhorn_tensor(ds4_gpu_tensor *out, const ds4_gpu_tensor *mix,
    const void *model_map, uint64_t model_size, uint64_t scale_offset,
    uint64_t base_offset, uint32_t n_hc, uint32_t sinkhorn_iters, float eps) {
    return dispatch_hc_split_weighted_sum(out, nullptr, out, mix, nullptr,
        model_map, model_size, scale_offset, base_offset, 0, 1, n_hc,
        sinkhorn_iters, eps, 0.0f, false, false);
}

int ds4_gpu_hc_split_weighted_sum_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *split,
    const ds4_gpu_tensor *mix, const ds4_gpu_tensor *rhc, const void *mm,
    uint64_t ms, uint64_t so, uint64_t bo, uint32_t n_embd, uint32_t n_hc,
    uint32_t si, float eps)
{
    return dispatch_hc_split_weighted_sum(out, nullptr, split, mix, rhc, mm, ms,
        so, bo, 0, n_embd, n_hc, si, eps, 0.0f, true, false);
}

int ds4_gpu_hc_split_weighted_sum_norm_tensor(ds4_gpu_tensor *out,
    ds4_gpu_tensor *norm_out, ds4_gpu_tensor *split, const ds4_gpu_tensor *mix,
    const ds4_gpu_tensor *rhc, const void *mm, uint64_t ms, uint64_t so,
    uint64_t bo, uint64_t norm_weight_offset, uint32_t n_embd, uint32_t n_hc,
    uint32_t si, float eps, float norm_eps) {
    return dispatch_hc_split_weighted_sum(out, norm_out, split, mix, rhc, mm, ms,
        so, bo, norm_weight_offset, n_embd, n_hc, si, eps, norm_eps, true, true);
}

int ds4_gpu_hc_expand_split_tensor(ds4_gpu_tensor *out_hc, const ds4_gpu_tensor *block_out,
    const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
    uint32_t n_embd, uint32_t n_hc)
{
    if (!out_hc || !block_out || !residual_hc || !split || n_embd == 0 || n_hc == 0) return 0;
    const uint64_t hc_values = (uint64_t)n_hc * n_embd;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    if (hc_values == 0 || mix_hc == 0) return 0;
    uint64_t rows = out_hc->bytes / (hc_values * sizeof(float));
    rows = std::min(rows, block_out->bytes / ((uint64_t)n_embd * sizeof(float)));
    rows = std::min(rows, residual_hc->bytes / (hc_values * sizeof(float)));
    rows = std::min(rows, split->bytes / (mix_hc * sizeof(float)));
    return rows > UINT32_MAX ? 0 : dispatch_hc_expand(out_hc, block_out, nullptr,
        residual_hc, split, split, n_embd, n_hc, (uint32_t)rows, true, false, false);
}

int ds4_gpu_hc_expand_add_split_tensor(ds4_gpu_tensor *out_hc,
    const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add,
    const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
    uint32_t n_embd, uint32_t n_hc)
{
    if (!out_hc || !block_out || !block_add || !residual_hc || !split ||
        n_embd == 0 || n_hc == 0) return 0;
    const uint64_t hc_values = (uint64_t)n_hc * n_embd;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    if (hc_values == 0 || mix_hc == 0) return 0;
    const uint64_t hc_row_bytes = hc_values * sizeof(float);
    const uint64_t embd_row_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t split_row_bytes = mix_hc * sizeof(float);
    uint64_t rows = out_hc->bytes / hc_row_bytes;
    rows = std::min(rows, block_out->bytes / embd_row_bytes);
    rows = std::min(rows, block_add->bytes / embd_row_bytes);
    rows = std::min(rows, residual_hc->bytes / hc_row_bytes);
    rows = std::min(rows, split->bytes / split_row_bytes);
    return rows > UINT32_MAX ? 0 : dispatch_hc_expand(out_hc, block_out, block_add,
        residual_hc, split, split, n_embd, n_hc, (uint32_t)rows, true, false, false);
}

/* ---- HC expand (single-token) ----
 *
 * ds4.c hc_post_one: inject the sublayer output into every HC stream and
 * mix the previous HC streams through the learned combine matrix.
 *   out_hc[dst*n_embd + d] = block_out[d] * post[dst]
 *                            + sum_src comb[dst + src*n_hc] * residual_hc[src*n_embd + d]
 * where post has n_hc gates and comb is the n_hc x n_hc combine matrix
 * addressed [dst_hc, src_hc] (row dst, column src). */
int ds4_gpu_hc_expand_tensor(ds4_gpu_tensor *out_hc,
    const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *residual_hc,
    const ds4_gpu_tensor *post, const ds4_gpu_tensor *comb,
    uint32_t n_embd, uint32_t n_hc)
{
    if (!out_hc || !block_out || !residual_hc || !post || !comb ||
        n_embd == 0 || n_hc == 0) return 0;
    const uint64_t out_row = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t block_row = (uint64_t)n_embd * sizeof(float);
    const uint64_t residual_row = out_row;
    const uint64_t post_row = (uint64_t)n_hc * sizeof(float);
    const uint64_t comb_row = (uint64_t)n_hc * n_hc * sizeof(float);
    uint64_t rows = std::min(out_hc->bytes / out_row, block_out->bytes / block_row);
    rows = std::min(rows, residual_hc->bytes / residual_row);
    rows = std::min(rows, post->bytes / post_row);
    rows = std::min(rows, comb->bytes / comb_row);
    return rows > UINT32_MAX ? 0 : dispatch_hc_expand(out_hc, block_out, nullptr,
        residual_hc, post, comb, n_embd, n_hc, (uint32_t)rows, false, false, false);
}

/* ---- Fused Q8_0 matmul + HC expand ----
 *
 * ds4_gpu_matmul_q8_0_pair_tensor: two Q8_0 projections with separate
 * weight matrices in one call (engine: q+kv or gate+up paired projections).
 *   out0[t][o] = sum_i x[t][i] * W0[o][i]
 *   out1[t][o] = sum_i x[t][i] * W1[o][i]
 * with the same Q8_0 dequant math as the verified matmul_q8_0 shader
 * (f16 block scale, int8 quants, raw f32 activations, double accumulation).
 *
 * ds4_gpu_matmul_q8_0_hc_expand_tensor: fused attention-output projection +
 * hc_post_one (single token, decode): block_out = W @ x, then
 *   out_hc[dst*n_embd + d] = block_out[d] * post[dst]
 *                            + sum_src comb[dst + src*n_hc] * residual_hc[src*n_embd + d]
 * with post = split[n_hc .. 2*n_hc), comb = split[2*n_hc .. 2*n_hc + n_hc*n_hc)
 * (the engine's hc_split sinkhorn layout, matching the Metal fused kernel).
 *
 * The fused operation is composed from the real Vulkan matmul and HC expand
 * dispatches so both output tensors retain their normal GPU synchronization.
 */

int ds4_gpu_matmul_q8_0_pair_tensor(
    ds4_gpu_tensor *out0, ds4_gpu_tensor *out1,
    const void *model_map, uint64_t model_size,
    uint64_t weight0_offset, uint64_t weight1_offset,
    uint64_t in_dim, uint64_t out0_dim, uint64_t out1_dim,
    const ds4_gpu_tensor *x, uint64_t n_tok)
{
    if (!out0 || !out1 || !model_map || !x || in_dim == 0 || n_tok == 0 ||
        out0_dim == 0 || out1_dim == 0) return 0;
    if (!out0->ptr || !out1->ptr || !x->ptr) return 0;

    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    /* Never read past the model mmap (SIGBUS guard), never write past
     * tensor bytes. */
    if (weight0_offset > model_size ||
        out0_dim > (model_size - weight0_offset) / row_bytes) return 0;
    if (weight1_offset > model_size ||
        out1_dim > (model_size - weight1_offset) / row_bytes) return 0;
    if (n_tok > UINT64_MAX / in_dim ||
        (uint64_t)in_dim * n_tok * sizeof(float) > x->bytes) return 0;
    if (n_tok > UINT64_MAX / out0_dim ||
        (uint64_t)out0_dim * n_tok * sizeof(float) > out0->bytes) return 0;
    if (n_tok > UINT64_MAX / out1_dim ||
        (uint64_t)out1_dim * n_tok * sizeof(float) > out1->bytes) return 0;

    /* Both projections consume the same activation.  The ordinary Q8 path
     * quantizes that activation into a temporary for each call, which turns
     * a pair into two quantize dispatches and two temporary buffers.  Keep
     * one quantized activation alive across both matmuls instead.  This is
     * deliberately limited to the same capability/shape envelope as the
     * prequant path; all other shapes retain the established fallback below.
     *
     * In a layer batch submit_and_wait() is intentionally a no-op, and
     * ds4_gpu_tensor_free() defers the temporary until the batch retires. In
     * the unbatched path the explicit wait below makes the lifetime safe. */
    const bool prequant_eligible =
        n_tok <= 65535u && blocks <= 256u &&
        n_tok <= UINT64_MAX / blocks &&
        n_tok * blocks <= UINT64_MAX / 36u &&
        n_tok * blocks * 36u <= UINT32_MAX &&
        g_vk.caps.max_compute_work_group_size[0] >= 256u &&
        g_vk.caps.max_compute_work_group_invocations >= 256u;
    if (prequant_eligible) {
        ds4_gpu_tensor *q = ds4_gpu_tensor_alloc_device_scratch(
            n_tok * blocks * 36u);
        if (!q) return 0;
        int ok = ds4_gpu_quantize_q8_0_tensor(q, x, in_dim, n_tok);
        if (ok)
            ok = ds4_gpu_matmul_q8_0_prequant_tensor(
                out0, model_map, model_size, weight0_offset,
                in_dim, out0_dim, q, n_tok);
        if (ok)
            ok = ds4_gpu_matmul_q8_0_prequant_tensor(
                out1, model_map, model_size, weight1_offset,
                in_dim, out1_dim, q, n_tok);

        auto &ctx = get_cmd_ctx();
        if (ctx.attention_output_batch) {
            free_or_defer_attention_tensor(ctx, q);
            return ok;
        }
        if (ok) {
            /* The temporary is referenced by both recorded matmuls. */
            ok = submit_and_wait();
        } else if (ctx.recording && ctx.command_count != 0) {
            submit_and_wait();
        }
        ds4_gpu_tensor_free(q);
        return ok;
    }

    if (ds4_gpu_matmul_q8_0_tensor(out0, model_map, model_size,
                                   weight0_offset, in_dim, out0_dim,
                                   x, n_tok) == 0) return 0;
    return ds4_gpu_matmul_q8_0_tensor(out1, model_map, model_size,
                                      weight1_offset, in_dim, out1_dim,
                                      x, n_tok);
}

int ds4_gpu_matmul_q8_0_hc_expand_tensor(
    ds4_gpu_tensor *out_hc, ds4_gpu_tensor *block_out,
    const void *model_map, uint64_t model_size, uint64_t weight_offset,
    uint64_t in_dim, uint64_t out_dim, const ds4_gpu_tensor *x,
    const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
    uint32_t n_embd, uint32_t n_hc)
{
    if (!out_hc || !block_out || !model_map || !x || !residual_hc || !split ||
        n_embd == 0 || n_hc == 0 || in_dim == 0 || out_dim != n_embd) return 0;
    if (!out_hc->ptr || !block_out->ptr || !x->ptr ||
        !residual_hc->ptr || !split->ptr) return 0;

    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * 34u;
    if (weight_offset > model_size ||
        out_dim > (model_size - weight_offset) / row_bytes) return 0;

    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    if ((uint64_t)in_dim * sizeof(float) > x->bytes ||
        embd_bytes > block_out->bytes ||
        hc_bytes > residual_hc->bytes ||
        hc_bytes > out_hc->bytes ||
        mix_hc * sizeof(float) > split->bytes) return 0;

    /* The production Flash attention-output-B projection is 4096 -> 4096
     * with four HC streams and 128 Q8 blocks per row.  In that shape the
     * composed implementation below does two full dispatches and writes /
     * rereads a 16 KiB block_out tensor.  Keep the old path as the exact
     * fallback, but let the appliance kernel compute each two-row projection
     * and immediately apply HC post/comb in the same workgroup. */
    const bool fuse_rows2 =
        in_dim == 4096u && out_dim == 4096u && n_embd == 4096u &&
        n_hc == 4u && blocks == 128u &&
        getenv("DS4_VULKAN_DISABLE_Q8_HC_EXPAND_FUSE") == nullptr;
    if (fuse_rows2) {
        auto shader_it = g_vk.shader_map.find("matmul_q8_0_hc_expand_rows2_bfe");
        decltype(g_vk.aligned_cache)::mapped_type *aligned = nullptr;
        const bool aligned_ok = shader_it != g_vk.shader_map.end() &&
            ensure_aligned_weight(model_map, model_size, weight_offset,
                                  in_dim, out_dim, aligned) && aligned;
        const uint64_t q_bytes = blocks * 36u;
        const uint64_t records = out_dim * blocks;
        const uint64_t scale_bytes = (records * 2u + 3u) & ~3ull;
        const uint64_t payload_bytes = records * 32u;
        if (aligned_ok && q_bytes <= UINT32_MAX &&
            scale_bytes <= aligned->scale_bytes &&
            payload_bytes <= aligned->payload_bytes &&
            (!g_vk.caps.max_storage_buffer_range ||
             (scale_bytes <= g_vk.caps.max_storage_buffer_range &&
              payload_bytes <= g_vk.caps.max_storage_buffer_range))) {
            ds4_gpu_tensor *q = ds4_gpu_tensor_alloc_device_scratch(q_bytes);
            if (q) {
                auto &ctx = get_cmd_ctx();
                const bool resume_recording = ctx.recording;
                int ok = ds4_gpu_quantize_q8_0_tensor(q, x, in_dim, 1);
                VkBuffer xbuf, bbuf, hbuf, rbuf, sbuf;
                VkDeviceSize xoff, boff, hoff, roff, soff;
                if (ok && find_tensor_buffer(q, xbuf, xoff) &&
                    find_tensor_buffer(block_out, bbuf, boff) &&
                    find_tensor_buffer(out_hc, hbuf, hoff) &&
                    find_tensor_buffer(residual_hc, rbuf, roff) &&
                    find_tensor_buffer(split, sbuf, soff)) {
                    const VkDeviceSize align =
                        g_vk.caps.min_storage_buffer_offset_alignment;
                    if (align && ((xoff | boff | hoff | roff | soff) % align) != 0) {
                        ok = 0;
                    } else {
                        VkDescriptorBufferInfo buffers[7] = {
                            {xbuf, xoff, (VkDeviceSize)q_bytes},
                            {aligned->gpu.buffer, 0, (VkDeviceSize)scale_bytes},
                            {aligned->gpu.buffer, (VkDeviceSize)aligned->payload_offset,
                             (VkDeviceSize)payload_bytes},
                            {bbuf, boff, (VkDeviceSize)embd_bytes},
                            {hbuf, hoff, (VkDeviceSize)hc_bytes},
                            {rbuf, roff, (VkDeviceSize)hc_bytes},
                            {sbuf, soff, (VkDeviceSize)(mix_hc * sizeof(float))},
                        };
                        struct { uint32_t in_dim, out_dim, n_hc, blocks; } pc = {
                            (uint32_t)in_dim, (uint32_t)out_dim, n_hc,
                            (uint32_t)blocks};
                        ok = record_simple_shader(
                            "matmul_q8_0_hc_expand_rows2_bfe", &pc, sizeof(pc),
                            buffers, 7, (uint32_t)((out_dim + 1u) / 2u), 1, 1,
                            resume_recording);
                    }
                } else {
                    ok = 0;
                }
                /* The quantization and fused dispatch both reference q.  A
                 * non-batched call must retire before releasing it; layer
                 * batching defers the release through the normal tensor
                 * lifetime path. */
                if (ctx.layer_batch_active) {
                    ds4_gpu_tensor_free(q);
                } else {
                    if (ok) ok = submit_and_wait();
                    else if (ctx.recording && ctx.command_count != 0)
                        (void)submit_and_wait();
                    ds4_gpu_tensor_free(q);
                }
                if (ok) return 1;
                /* A missing/incompatible candidate must not break the
                 * production path: fall through to the established pair. */
            }
        }
    }

    if (ds4_gpu_matmul_q8_0_tensor(block_out, model_map, model_size,
                                   weight_offset, in_dim, out_dim, x, 1) == 0)
        return 0;

    return ds4_gpu_hc_expand_split_tensor(out_hc, block_out, residual_hc,
                                          split, n_embd, n_hc);
}

/* ---- HC expand-add (single-token) ----
 *
 * Same as ds4_gpu_hc_expand_tensor but the sublayer output is the sum of
 * two tensors (used when the attention output arrives split across tensor
 * parallel replicas, e.g. tp_attn_a + tp_attn_b):
 *   out_hc[dst*n_embd + d] = (block_out[d] + block_add[d]) * post[dst]
 *                            + sum_src comb[dst + src*n_hc] * residual_hc[src*n_embd + d] */
int ds4_gpu_hc_expand_add_tensor(ds4_gpu_tensor *out_hc,
    const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add,
    const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *post,
    const ds4_gpu_tensor *comb, uint32_t n_embd, uint32_t n_hc)
{
    if (!out_hc || !block_out || !block_add || !residual_hc || !post || !comb ||
        n_embd == 0 || n_hc == 0) return 0;
    const uint64_t out_row = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t block_row = (uint64_t)n_embd * sizeof(float);
    const uint64_t post_row = (uint64_t)n_hc * sizeof(float);
    const uint64_t comb_row = (uint64_t)n_hc * n_hc * sizeof(float);
    uint64_t rows = std::min(out_hc->bytes / out_row, block_out->bytes / block_row);
    rows = std::min(rows, block_add->bytes / block_row);
    rows = std::min(rows, residual_hc->bytes / out_row);
    rows = std::min(rows, post->bytes / post_row);
    rows = std::min(rows, comb->bytes / comb_row);
    return rows > UINT32_MAX ? 0 : dispatch_hc_expand(out_hc, block_out, block_add,
        residual_hc, post, comb, n_embd, n_hc, (uint32_t)rows, false, false, false);
}

/* ---- HC expand-split, f16 block (batch fast path) ----
 *
 * Same as ds4_gpu_hc_expand_split_tensor but the sublayer output arrives in
 * f16 (2 B/element). The split layout is [pre | post | comb], matching
 * hc_post_one and the CUDA implementation. */
int ds4_gpu_hc_expand_split_half_tensor(ds4_gpu_tensor *out_hc,
    const ds4_gpu_tensor *block_out_h, const ds4_gpu_tensor *residual_hc,
    const ds4_gpu_tensor *split, uint32_t n_embd, uint32_t n_hc)
{
    DS4_VK_TRACE_KERNEL("hc_expand_split_half");
    if (!out_hc || !block_out_h || !residual_hc || !split) return 0;
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t half_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t split_bytes = mix_hc * sizeof(float);
    if (hc_bytes == 0 || half_bytes == 0 || split_bytes == 0) return 0;
    uint64_t rows = out_hc->bytes / hc_bytes;
    rows = std::min(rows, block_out_h->bytes / half_bytes);
    rows = std::min(rows, residual_hc->bytes / hc_bytes);
    rows = std::min(rows, split->bytes / split_bytes);
    return rows > UINT32_MAX ? 0 : dispatch_hc_expand(out_hc, block_out_h, nullptr,
        residual_hc, split, split, n_embd, n_hc, (uint32_t)rows, true, true, false);
}

/* ---- HC expand-add-split, f16 add block (batch fast path) ----
 *
 * Same as ds4_gpu_hc_expand_add_split_tensor but the second block arrives
 * in f16 (2 B/element): the engine's shared_down_f16 FFN path feeds the
 * shared-expert half from g->batch_q_half.  The split layout is the
 * sinkhorn buffer [pre | post | comb], matching hc_post_one. */
int ds4_gpu_hc_expand_add_split_half_add_tensor(ds4_gpu_tensor *out_hc,
    const ds4_gpu_tensor *block_out, const ds4_gpu_tensor *block_add_h,
    const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *split,
    uint32_t n_embd, uint32_t n_hc)
{
    DS4_VK_TRACE_KERNEL("hc_expand_add_split_half_add");
    if (!out_hc || !block_out || !block_add_h || !residual_hc || !split) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr,
                    "ds4: [dbg] hc_expand_add_split_half_add missing tensor "
                    "out=%p block=%p half=%p residual=%p split=%p\n",
                    (void *)out_hc, (const void *)block_out,
                    (const void *)block_add_h, (const void *)residual_hc,
                    (const void *)split);
        return 0;
    }
    const uint64_t hc_bytes = (uint64_t)n_hc * n_embd * sizeof(float);
    const uint64_t embd_bytes = (uint64_t)n_embd * sizeof(float);
    const uint64_t half_bytes = (uint64_t)n_embd * sizeof(uint16_t);
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    const uint64_t split_bytes = mix_hc * sizeof(float);
    if (hc_bytes == 0 || embd_bytes == 0 || half_bytes == 0 || split_bytes == 0) return 0;
    uint64_t rows = out_hc->bytes / hc_bytes;
    rows = std::min(rows, block_out->bytes / embd_bytes);
    rows = std::min(rows, block_add_h->bytes / half_bytes);
    rows = std::min(rows, residual_hc->bytes / hc_bytes);
    rows = std::min(rows, split->bytes / split_bytes);
    if (getenv("DS4_VULKAN_DEBUG"))
        fprintf(stderr,
                "ds4: [dbg] hc_expand_add_split_half_add shape "
                "embd=%u hc=%u rows=%llu out=%llu block=%llu half=%llu "
                "residual=%llu split=%llu\n",
                n_embd, n_hc, (unsigned long long)rows,
                (unsigned long long)out_hc->bytes,
                (unsigned long long)block_out->bytes,
                (unsigned long long)block_add_h->bytes,
                (unsigned long long)residual_hc->bytes,
                (unsigned long long)split->bytes);
    if (rows == 0) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr,
                    "ds4: [dbg] hc_expand_add_split_half_add has no complete row "
                    "out=%llu block=%llu half=%llu residual=%llu split=%llu\n",
                    (unsigned long long)out_hc->bytes,
                    (unsigned long long)block_out->bytes,
                    (unsigned long long)block_add_h->bytes,
                    (unsigned long long)residual_hc->bytes,
                    (unsigned long long)split->bytes);
        return 0;
    }
    int result = rows > UINT32_MAX ? 0 : dispatch_hc_expand(out_hc, block_out,
        block_add_h, residual_hc, split, split, n_embd, n_hc, (uint32_t)rows,
        true, false, true);
    if (getenv("DS4_VULKAN_DEBUG"))
        fprintf(stderr,
                "ds4: [dbg] hc_expand_add_split_half_add complete rows=%llu\n",
                (unsigned long long)rows);
    return result;
}

/* ---- HC weighted sum (single-token) ----
 *
 * ds4.c hc_weighted_sum_one: out[d] = sum_h residual_hc[h*n_embd + d] * weights[h].
 * `weights` is the sinkhorn split output; only the first n_hc entries (the
 * pre weights) participate, matching the engine's decode and output paths. */
int ds4_gpu_hc_weighted_sum_tensor(ds4_gpu_tensor *out,
    const ds4_gpu_tensor *residual_hc, const ds4_gpu_tensor *weights,
    uint32_t n_embd, uint32_t n_hc)
{
    if (!out || !residual_hc || !weights) return 0;
    if ((uint64_t)n_embd * sizeof(float) > out->bytes) return 0;
    if ((uint64_t)n_hc * n_embd * sizeof(float) > residual_hc->bytes) return 0;
    if ((uint64_t)n_hc * sizeof(float) > weights->bytes) return 0;
    return dispatch_hc_weighted_sum(out, residual_hc, weights, n_embd, n_hc, 1);
}

static int dispatch_output_hc_weights(ds4_gpu_tensor *out,
    const ds4_gpu_tensor *pre, uint64_t scale_offset, uint64_t base_offset,
    uint32_t n_hc, uint32_t n_tokens, float eps) {
    auto si = g_vk.shader_map.find("output_hc_weights");
    if (si == g_vk.shader_map.end()) return 0;

    VkBuffer obuf, pbuf, scale_buf, base_buf;
    VkDeviceSize ooff, poff, scale_off, base_off;
    if (!find_tensor_buffer(out, obuf, ooff) ||
        !find_tensor_buffer(pre, pbuf, poff) ||
        !hc_cached_weight(scale_offset, sizeof(float), scale_buf, scale_off) ||
        !hc_cached_weight(base_offset, (uint64_t)n_hc * sizeof(float),
                          base_buf, base_off)) return 0;

    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;

    auto &shader = g_vk.shaders[si->second];
    VkDescriptorBufferInfo buffers[4] = {
        {obuf, ooff, (VkDeviceSize)n_tokens * n_hc * sizeof(float)},
        {pbuf, poff, (VkDeviceSize)n_tokens * n_hc * sizeof(float)},
        {scale_buf, scale_off, sizeof(float)},
        {base_buf, base_off, (VkDeviceSize)n_hc * sizeof(float)},
    };
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, 4, set))
        return fail_simple_dispatch(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    struct { uint32_t n_hc, n_tokens; float eps; uint32_t reserved; } push = {
        n_hc, n_tokens, eps, 0};
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    timeline_dispatch(ctx, "output_hc_weights", buffers, 4,
                      (n_hc + 255u) / 256u, n_tokens, 1);
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_simple_descriptors(set)) ok = 0;
    return ok;
}

/* ---- Output-head HC weights (single-token) ----
 *
 * ds4.c output_hc_head_one: out[i] = sigmoid_stable(pre[i] * scale[0] + base[i]) + eps,
 * with a scalar model scale and an n_hc-wide model base.  These become the
 * weights of the final hc_weighted_sum before the vocab projection. */
int ds4_gpu_output_hc_weights_tensor(ds4_gpu_tensor *out,
    const ds4_gpu_tensor *pre, const void *model_map, uint64_t model_size,
    uint64_t scale_offset, uint64_t base_offset, uint32_t n_hc, float eps)
{
    if (!out || !pre || !model_map || n_hc == 0) return 0;
    const uint64_t row_bytes = (uint64_t)n_hc * sizeof(float);
    if (row_bytes == 0 || out->bytes < row_bytes || out->bytes % row_bytes != 0 ||
        pre->bytes < out->bytes) return 0;
    if (scale_offset > model_size || sizeof(float) > model_size - scale_offset) return 0;
    if (base_offset > model_size || row_bytes > model_size - base_offset) return 0;
    const uint64_t n_tokens = out->bytes / row_bytes;
    if (n_tokens == 0 || n_tokens > 65535u) return 0;
    if (g_vk.model_map != model_map || g_vk.model_size != model_size)
        set_model_map_identity(model_map, model_size);
    return dispatch_output_hc_weights(out, pre, scale_offset, base_offset,
                                      n_hc, (uint32_t)n_tokens, eps);
}

static int dispatch_router_select(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    ds4_gpu_tensor *probs, const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens,
    VkBuffer bias_buf, VkDeviceSize bias_off, VkDeviceSize bias_bytes,
    VkBuffer hash_buf, VkDeviceSize hash_off, VkDeviceSize hash_bytes,
    uint32_t hash_rows, uint32_t token, bool has_bias, bool hash_mode,
    float scale, uint32_t n_tokens) {
    auto si = g_vk.shader_map.find("router_select");
    if (si == g_vk.shader_map.end()) return 0;
    VkBuffer selected_buf, weights_buf, probs_buf, logits_buf, tokens_buf;
    VkDeviceSize selected_off, weights_off, probs_off, logits_off, tokens_off;
    if (!find_tensor_buffer(selected, selected_buf, selected_off) ||
        !find_tensor_buffer(weights, weights_buf, weights_off) ||
        !find_tensor_buffer(probs, probs_buf, probs_off) ||
        !find_tensor_buffer(logits, logits_buf, logits_off) ||
        !find_tensor_buffer(tokens, tokens_buf, tokens_off)) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    for (auto &[base, entry] : g_vk.weight_cache) {
        (void)base;
        if (entry.buffer == bias_buf || entry.buffer == hash_buf)
            mark_weight_generation(entry);
    }
    const VkDeviceSize dummy_bytes = (VkDeviceSize)logits->bytes;
    VkDescriptorBufferInfo buffers[7] = {
        {selected_buf, selected_off, (VkDeviceSize)selected->bytes},
        {weights_buf, weights_off, (VkDeviceSize)weights->bytes},
        {probs_buf, probs_off, (VkDeviceSize)probs->bytes},
        {logits_buf, logits_off, dummy_bytes},
        {has_bias ? bias_buf : logits_buf, has_bias ? bias_off : logits_off,
         has_bias ? bias_bytes : dummy_bytes},
        {hash_mode ? hash_buf : logits_buf, hash_mode ? hash_off : logits_off,
         hash_mode ? hash_bytes : dummy_bytes},
        {tokens_buf, tokens_off, (VkDeviceSize)tokens->bytes},
    };
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, 7, set))
        return fail_simple_dispatch(ctx);
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    struct { uint32_t n_tokens, hash_rows, token, has_bias, hash_mode; float scale; } push = {
        n_tokens, hash_rows, token, has_bias ? 1u : 0u, hash_mode ? 1u : 0u, scale};
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(push), &push);
    timeline_dispatch(ctx, "router_select", buffers, 7, n_tokens, 1, 1);
    int ok = finish_simple_dispatch(ctx, resume_recording, true);
    if (!release_simple_descriptors(set)) ok = 0;
    return ok;
}

int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size,
    uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows,
    uint32_t token, uint32_t n_expert, uint32_t n_expert_used,
    float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used,
    bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits)
{
    if (n_expert != 256u || n_expert_used != 6u ||
        fabsf(expert_weight_scale - 1.5f) > 1.0e-6f ||
        n_expert_groups > 1u || n_group_used > 0u) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        !selected->ptr || !weights->ptr || !probs->ptr || !logits->ptr) return 0;
    /* Decode supplies one router-logits row; hash routing uses the token ID. */
    const uint64_t prob_bytes = 256u * sizeof(float);
    const uint64_t selected_bytes = 6u * sizeof(int32_t);
    const uint64_t hash_bytes = (uint64_t)hash_rows * selected_bytes;
    if ((hash_mode && hash_rows == 0) || logits->bytes < prob_bytes ||
        probs->bytes < prob_bytes || selected->bytes < selected_bytes ||
        weights->bytes < 6u * sizeof(float)) return 0;
    if (g_vk.model_map != model_map || g_vk.model_size != model_size)
        set_model_map_identity(model_map, model_size);
    VkBuffer bias_buf = VK_NULL_HANDLE, hash_buf = VK_NULL_HANDLE;
    VkDeviceSize bias_off = 0, hash_off = 0;
    if (has_bias && !hash_mode &&
        (bias_offset > model_size || prob_bytes > model_size - bias_offset ||
         !hc_cached_weight(bias_offset, prob_bytes, bias_buf, bias_off))) return 0;
    if (hash_mode &&
        (hash_offset > model_size || hash_bytes > model_size - hash_offset ||
         !hc_cached_weight(hash_offset, hash_bytes, hash_buf, hash_off))) return 0;
    ds4_gpu_tensor *token_tensor = ds4_gpu_tensor_alloc(sizeof(int32_t));
    if (!token_tensor || !ds4_gpu_tensor_write(token_tensor, 0, &token, sizeof(token))) {
        if (token_tensor) ds4_gpu_tensor_free(token_tensor);
        return 0;
    }
    int ok = dispatch_router_select(selected, weights, probs, logits, token_tensor,
        bias_buf, bias_off, prob_bytes, hash_buf, hash_off, hash_bytes,
        hash_rows, token, has_bias && !hash_mode, hash_mode,
        expert_weight_scale, 1);
    ds4_gpu_tensor_free(token_tensor);
    return ok;
}

int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    ds4_gpu_tensor *probs, const void *mm, uint64_t ms, uint64_t bo, uint64_t ho,
    uint32_t hr, uint32_t ng, uint32_t ngu, bool hb, bool hm,
    const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens,
    uint32_t ne, uint32_t neu, float ws, uint32_t nt)
{
    if (ne != 256u || neu != 6u || fabsf(ws - 1.5f) > 1.0e-6f ||
        ng > 1u || ngu > 0u) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !mm ||
        nt == 0 || nt > 65535u ||
        !selected->ptr || !weights->ptr || !probs->ptr ||
        !logits->ptr || !tokens->ptr) return 0;
    const uint64_t prob_bytes = 256u * sizeof(float);
    const uint64_t logits_bytes = (uint64_t)nt * 256u * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)nt * 6u * sizeof(int32_t);
    const uint64_t hash_bytes = (uint64_t)hr * 6u * sizeof(int32_t);
    if (logits_bytes > logits->bytes || logits_bytes > probs->bytes ||
        selected_bytes > selected->bytes || selected_bytes > weights->bytes ||
        (uint64_t)nt * sizeof(int32_t) > tokens->bytes || (hm && hr == 0) ||
        tokens->bytes < (uint64_t)nt * sizeof(int32_t)) return 0;
    if (g_vk.model_map != mm || g_vk.model_size != ms) set_model_map_identity(mm, ms);
    VkBuffer bias_buf = VK_NULL_HANDLE, hash_buf = VK_NULL_HANDLE;
    VkDeviceSize bias_off = 0, hash_off = 0;
    if (hb && !hm && (bo > ms || prob_bytes > ms - bo ||
        !hc_cached_weight(bo, prob_bytes, bias_buf, bias_off))) return 0;
    if (hm && (ho > ms || hash_bytes > ms - ho ||
        !hc_cached_weight(ho, hash_bytes, hash_buf, hash_off))) return 0;
    return dispatch_router_select(selected, weights, probs, logits, tokens,
        bias_buf, bias_off, prob_bytes, hash_buf, hash_off, hash_bytes,
        hr, 0, hb && !hm, hm, ws, nt);
}

/* ---- DS4 indexer: compressed-row selection (Vulkan compute) ----
 *
 * Replicates the engine's CPU reference (ds4.c indexer_allowed_decode_one*
 * and the per-token score loop) and the Metal/CUDA indexer score kernels
 * exactly:
 *
 *     score[c] = sum_h max(0, dot(q[h], index_comp[c])) * weights[h] * scale
 *
 * Layouts (all f32, row-major):
 *   q          [n_head][head_dim]             (single-token decode)
 *   weights    [n_head]
 *   index_comp [n_comp][head_dim]
 *   scores     [n_comp]
 *
 * The batched decode variant uses q [n_tokens][n_head][head_dim],
 * weights [n_tokens][n_head] and scores [n_tokens][n_comp].  Like the
 * Metal tiled kernel it applies the causal visibility mask: at position
 * p = pos0 + t only the first (p + 1) / ratio compressed rows exist, so
 * rows beyond that are written as -INFINITY (the top-k selection below
 * then ignores them).
 */

static int record_indexer_scores(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t pos0,
        uint32_t n_head, uint32_t head_dim, uint32_t ratio,
        float scale, uint32_t causal) {
    if (!scores || !q || !weights || !index_comp || n_comp == 0 ||
        n_tokens == 0 || n_head == 0 || head_dim == 0 || ratio == 0 ||
        n_comp > 4096u || n_tokens > 65535u || n_head > 65535u ||
        (causal && n_tokens > 1 && pos0 > UINT32_MAX - (n_tokens - 1u)) ||
        head_dim > 128u || !shader_f32_domain(n_comp, n_tokens, 1) ||
        !shader_f32_domain(n_tokens, n_head, head_dim) ||
        !shader_f32_domain(n_comp, 1, head_dim)) return 0;
    uint64_t score_bytes, q_bytes, weight_bytes, key_bytes;
    if (!checked_f32_bytes(n_tokens, n_comp, 1, score_bytes) ||
        !checked_f32_bytes(n_tokens, n_head, head_dim, q_bytes) ||
        !checked_f32_bytes(n_tokens, n_head, 1, weight_bytes) ||
        !checked_f32_bytes(n_comp, 1, head_dim, key_bytes) ||
        scores->bytes < score_bytes || q->bytes < q_bytes ||
        weights->bytes < weight_bytes || index_comp->bytes < key_bytes)
        return 0;
    VkBuffer sbuf, qbuf, wbuf, kbuf;
    VkDeviceSize soff, qoff, woff, koff;
    if (!find_tensor_buffer(scores, sbuf, soff) || !find_tensor_buffer(q, qbuf, qoff) ||
        !find_tensor_buffer(weights, wbuf, woff) || !find_tensor_buffer(index_comp, kbuf, koff))
        return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((soff | qoff | woff | koff) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[4] = {
        {sbuf, soff, (VkDeviceSize)scores->bytes}, {qbuf, qoff, (VkDeviceSize)q->bytes},
        {wbuf, woff, (VkDeviceSize)weights->bytes}, {kbuf, koff, (VkDeviceSize)index_comp->bytes}};
    struct { uint32_t n_comp, n_tokens, pos0, n_head, head_dim, ratio, causal; float scale; }
        pc = {n_comp, n_tokens, pos0, n_head, head_dim, ratio, causal, scale};
    DS4_VK_TRACE_KERNEL("indexer_scores");
    return record_simple_shader("indexer_scores", &pc, sizeof(pc), bufs, 4,
                                n_comp, n_tokens, 1, resume_recording);
}

int ds4_gpu_indexer_score_one_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_head,
        uint32_t                head_dim,
        float                   scale) {
    return record_indexer_scores(scores, q, weights, index_comp,
                                 n_comp, 1, 0, n_head, head_dim, 1, scale, 0);
}

int ds4_gpu_indexer_scores_prefill_tensor(
        ds4_gpu_tensor *scores, const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights, const ds4_gpu_tensor *index_comp,
        uint32_t n_comp, uint32_t n_tokens, uint32_t n_head,
        uint32_t head_dim, uint32_t ratio, float scale) {
    return record_indexer_scores(scores, q, weights, index_comp, n_comp,
                                 n_tokens, 0, n_head, head_dim, ratio, scale, 1);
}

int ds4_gpu_indexer_scores_decode_batch_tensor(
        ds4_gpu_tensor       *scores,
        const ds4_gpu_tensor *q,
        const ds4_gpu_tensor *weights,
        const ds4_gpu_tensor *index_comp,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                pos0,
        uint32_t                n_head,
        uint32_t                head_dim,
        uint32_t                ratio,
        float                   scale) {
    return record_indexer_scores(scores, q, weights, index_comp,
                                 n_comp, n_tokens, pos0, n_head, head_dim,
                                 ratio, scale, 1);
}

int ds4_gpu_indexer_topk_tensor(
        ds4_gpu_tensor       *selected,
        const ds4_gpu_tensor *scores,
        uint32_t                n_comp,
        uint32_t                n_tokens,
        uint32_t                top_k) {
    if (!selected || !scores || n_comp == 0 || n_tokens == 0 ||
        top_k == 0 || top_k > n_comp || top_k > 512u || n_comp > 4096u ||
        n_tokens > 65535u) return 0;
    const uint64_t score_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)top_k * n_tokens * sizeof(uint32_t);
    if (scores->bytes < score_bytes || selected->bytes < selected_bytes) return 0;
    VkBuffer obuf, sbuf; VkDeviceSize ooff, soff;
    if (!find_tensor_buffer(selected, obuf, ooff) || !find_tensor_buffer(scores, sbuf, soff)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((ooff | soff) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx(); const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[2] = {{obuf, ooff, (VkDeviceSize)selected->bytes},
                                      {sbuf, soff, (VkDeviceSize)scores->bytes}};
    struct { uint32_t n_comp, n_tokens, top_k; } pc = {n_comp, n_tokens, top_k};
    DS4_VK_TRACE_KERNEL("indexer_topk");
    return record_simple_shader("indexer_topk", &pc, sizeof(pc), bufs, 2,
                                n_tokens, 1, 1, resume_recording);
}

int ds4_gpu_dsv4_topk_mask_tensor(
        ds4_gpu_tensor *mask, const ds4_gpu_tensor *topk,
        uint32_t n_comp, uint32_t n_tokens, uint32_t top_k) {
    if (!mask || !topk || n_comp == 0 || n_tokens == 0 || top_k == 0 ||
        top_k > n_comp || top_k > 512u || n_comp > 4096u || n_tokens > 65535u)
        return 0;
    const uint64_t mask_bytes = (uint64_t)n_comp * n_tokens * sizeof(float);
    const uint64_t topk_bytes = (uint64_t)top_k * n_tokens * sizeof(uint32_t);
    if (mask->bytes < mask_bytes || topk->bytes < topk_bytes) return 0;
    VkBuffer mbuf, tbuf; VkDeviceSize moff, toff;
    if (!find_tensor_buffer(mask, mbuf, moff) || !find_tensor_buffer(topk, tbuf, toff)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((moff | toff) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx(); const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[2] = {{mbuf, moff, (VkDeviceSize)mask->bytes},
                                      {tbuf, toff, (VkDeviceSize)topk->bytes}};
    struct { uint32_t n_comp, n_tokens, top_k; } pc = {n_comp, n_tokens, top_k};
    DS4_VK_TRACE_KERNEL("topk_mask");
    return record_simple_shader("topk_mask", &pc, sizeof(pc), bufs, 2,
                                n_tokens, 1, 1, resume_recording);
}

/* =========================================================================
 * DSV4 indexer QAT: in-place 128-wide Hadamard rotation + FP4
 * activation-simulation round trip on every row of x.
 *
 * Replicates ds4.c dsv4_indexer_qat_row_inplace_cpu
 * (dsv4_hadamard128_inplace_cpu + dsv4_fp4_act_quantize_row_inplace_cpu),
 * the CUDA indexer_hadamard_fp4_kernel and the Metal
 * kernel_dsv4_indexer_hadamard_fp4_f32.  head_dim is fixed at 128 by the
 * model graph (DS4_N_INDEXER_HEAD_DIM); the reference dies for any other
 * value, so invalid head_dim fails the call here too.
 *
 * FP4 activation quantization: per 32-element block, amax = max |x| (clamped
 * to a tiny floor), scale = 2^ceil(log2(amax/6)), each element is divided by
 * scale, clamped to [-6, 6], rounded through the E2M1FN values
 * {0, .5, 1, 1.5, 2, 3, 4, 6} (ties to even) and re-multiplied by scale.
 * ========================================================================= */

static float dsv4_e2m1fn_value_vk(int i) {
    static const float values[8] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    };
    return values[i & 7];
}

static float dsv4_e2m1fn_dequant_vk(float x) {
    const float sign = x < 0.0f ? -1.0f : 1.0f;
    const float ax = fminf(fabsf(x), 6.0f);
    int best = 0;
    float best_diff = fabsf(ax - dsv4_e2m1fn_value_vk(0));
    for (int i = 1; i < 8; i++) {
        const float diff = fabsf(ax - dsv4_e2m1fn_value_vk(i));
        if (diff < best_diff || (diff == best_diff && (i & 1) == 0 && (best & 1) != 0)) {
            best = i;
            best_diff = diff;
        }
    }
    return sign * dsv4_e2m1fn_value_vk(best);
}

static void dsv4_hadamard128_inplace_vk(float *x) {
    for (uint32_t stride = 1; stride < 128; stride <<= 1) {
        for (uint32_t base = 0; base < 128; base += 2u * stride) {
            for (uint32_t i = 0; i < stride; i++) {
                const float a = x[base + i];
                const float b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
        }
    }
    const float scale = 0.08838834764831845f;
    for (uint32_t i = 0; i < 128; i++) x[i] *= scale;
}

static void dsv4_fp4_act_quantize_row_inplace_vk(float *x, uint32_t n) {
    for (uint32_t off = 0; off < n; off += 32) {
        float amax = 0.0f;
        for (uint32_t i = 0; i < 32; i++) {
            const float av = fabsf(x[off + i]);
            if (av > amax) amax = av;
        }
        if (amax < 7.052966104933725e-38f) amax = 7.052966104933725e-38f;
        const float scale = ldexpf(1.0f, (int)ceilf(log2f(amax / 6.0f)));
        for (uint32_t i = 0; i < 32; i++) {
            float v = x[off + i] / scale;
            if (v > 6.0f) v = 6.0f;
            if (v < -6.0f) v = -6.0f;
            x[off + i] = dsv4_e2m1fn_dequant_vk(v) * scale;
        }
    }
}

int ds4_gpu_dsv4_indexer_qat_tensor(
        ds4_gpu_tensor *x,
        uint32_t          n_rows,
        uint32_t          head_dim) {
    DS4_VK_TRACE_KERNEL("dsv4_indexer_qat");
    if (!x || n_rows == 0 || head_dim != 128u) return 0;
    const uint64_t bytes = (uint64_t)n_rows * head_dim * sizeof(float);
    if (x->bytes < bytes) return 0;
    VkBuffer xbuf; VkDeviceSize xoff;
    if (!find_tensor_buffer(x, xbuf, xoff)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && xoff % align != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    constexpr uint32_t max_rows_per_dispatch = 16384u;
    const VkDeviceSize row_bytes = (VkDeviceSize)head_dim * sizeof(float);
    int ok = 1;
    for (uint32_t row_base = 0; row_base < n_rows; ) {
        const uint32_t tile_rows = std::min(max_rows_per_dispatch,
                                            n_rows - row_base);
        if (!ctx.recording && !begin_cmd()) return 0;
        const VkDeviceSize tile_offset = xoff + (VkDeviceSize)row_base * row_bytes;
        if ((align && tile_offset % align != 0) ||
            tile_rows > UINT64_MAX / row_bytes)
            return fail_simple_dispatch(ctx);
        VkDescriptorBufferInfo bufs[1] = {
            {xbuf, tile_offset, (VkDeviceSize)tile_rows * row_bytes}};
        struct { uint32_t n_rows; } pc = {tile_rows};
        ok = record_simple_shader("indexer_qat", &pc, sizeof(pc), bufs, 1,
                                  tile_rows, 1, 1, resume_recording);
        if (!ok) return 0;
        row_base += tile_rows;
    }
    return ok;
}

} /* extern "C" close CPU fallbacks */

/* ---- MoE CPU implementation ---- */
#include "ds4_gpu.h"

/* Q2_K block dequantization */
#define QK_KMOE 256

/* ds4_gpu_routed_moe_batch_tensor (prefill routed MoE) is implemented after
 * ds4_gpu_routed_moe_one_tensor below: it loops the same per-token math over
 * n_tokens slots.  The old stub here zeroed the output; the real
 * implementation lives next to the single-token kernel it shares helpers
 * with. */

/* =========================================================================
 * Single-token routed MoE: ds4_gpu_routed_moe_one_tensor
 *
 * Host-side implementation of the engine's routed expert FFN step.  The
 * reference math lives in ds4.c (layer_routed_moe_one_prealloc and the
 * matvec_*_prequant helpers); this is a faithful CPU port that reads the
 * quantized expert weights straight from the model mmap and writes into the
 * host-mapped tensors.  The quant block layouts, IQ2 tables, and dot helpers
 * below are copied from ds4.c (GGUF formats) because ds4.c cannot be
 * included by the backend.
 *
 * Supported routed tensor types (DS4_TENSOR_* ids, ds4.c:2040):
 *   gate/up: Q8_0 (8), Q2_K (10), IQ2_XXS (16)
 *   down:    Q8_0 (8), Q2_K (10), IQ2_XXS (16)
 * ========================================================================= */
namespace {

constexpr uint32_t DS4GK_QK_K = 256;

/* GGUF quant block layouts (copied from ds4.c, GGUF format). */
struct ds4gk_block_q2_K {
    uint8_t  scales[DS4GK_QK_K / 16];
    uint8_t  qs[DS4GK_QK_K / 4];
    uint16_t d;
    uint16_t dmin;
};

struct ds4gk_block_iq2_xxs {
    uint16_t d;
    uint16_t qs[DS4GK_QK_K / 8];
};

struct ds4gk_block_q8_K {
    float   d;
    int8_t  qs[DS4GK_QK_K];
    int16_t bsums[DS4GK_QK_K / 16];
};

/* IQ2_XXS tables (copied from ds4.c). */
static const uint8_t ds4gk_kmask_iq2xs[8] = {
    1, 2, 4, 8, 16, 32, 64, 128
};

static const uint8_t ds4gk_ksigns_iq2xs[128] = {
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
};

static const uint64_t ds4gk_iq2xxs_grid[256] = {
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
};

static int8_t ds4gk_iq2xxs_signed_grid[256][128][8];
static std::once_flag ds4gk_iq2xxs_once;

static void ds4gk_iq2xxs_signed_grid_init(void) {
    for (uint32_t g = 0; g < 256; g++) {
        const uint8_t *grid = (const uint8_t *)(ds4gk_iq2xxs_grid + g);
        for (uint32_t s = 0; s < 128; s++) {
            const uint8_t signs = ds4gk_ksigns_iq2xs[s];
            for (uint32_t j = 0; j < 8; j++) {
                const int v = (int)grid[j];
                ds4gk_iq2xxs_signed_grid[g][s][j] =
                    (int8_t)((signs & ds4gk_kmask_iq2xs[j]) ? -v : v);
            }
        }
    }
}

static void ds4gk_iq2xxs_ensure(void) {
    std::call_once(ds4gk_iq2xxs_once, ds4gk_iq2xxs_signed_grid_init);
}

static bool ds4gk_routed_iq2_lut_ensure(void) {
    std::lock_guard<std::recursive_mutex> lock(g_vk.cmd_mutex);
    if (g_vk.routed_iq2_lut.ptr) return true;
    uint32_t words[544] = {};
    for (uint32_t i = 0; i < 256; i++) {
        words[i * 2] = (uint32_t)ds4gk_iq2xxs_grid[i];
        words[i * 2 + 1] = (uint32_t)(ds4gk_iq2xxs_grid[i] >> 32);
    }
    for (uint32_t i = 0; i < 128; i++)
        words[512 + i / 4] |= (uint32_t)ds4gk_ksigns_iq2xs[i] << (8 * (i & 3));
    ds4_gpu_tensor lut{};
    if (ds4_gpu_tensor_alloc_on(&lut, 0, sizeof(words)) != 0) return false;
    if (!ds4_gpu_tensor_write(&lut, 0, words, sizeof(words))) {
        ds4_gpu_tensor_free_in_place(&lut);
        return false;
    }
    g_vk.routed_iq2_lut = lut;
    return true;
}

/* IEEE half -> f32 (copied from ds4.c). */
static inline float ds4gk_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ff;
            bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static float ds4gk_sigmoid_stable(float x) {
    if (x >= 0.0f) {
        const float e = expf(-x);
        return 1.0f / (1.0f + e);
    } else {
        const float e = expf(x);
        return e / (1.0f + e);
    }
}

static float ds4gk_silu(float x) { return x * ds4gk_sigmoid_stable(x); }

/* Q8_0 activation quantization (copied from ds4.c). */
static void ds4gk_quantize_q8_0_activation(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31) / 32;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32;
        const uint64_t bn = n - i0 < 32 ? n - i0 : 32;
        float amax = 0.0f;
        for (uint64_t i = 0; i < bn; i++) {
            const float ax = fabsf(x[i0 + i]);
            if (ax > amax) amax = ax;
        }
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        scale[b] = d;
        for (uint64_t i = 0; i < bn; i++) {
            int v = (int)lrintf(x[i0 + i] * id);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[i0 + i] = (int8_t)v;
        }
        for (uint64_t i = bn; i < 32 && i0 + i < blocks * 32; i++) {
            xq[i0 + i] = 0;
        }
    }
}

static inline int32_t ds4gk_dot_i8_32(const int8_t *a, const int8_t *b, uint64_t n) {
    int32_t sum = 0;
    for (uint64_t i = 0; i < n; i++) sum += (int32_t)a[i] * (int32_t)b[i];
    return sum;
}

/* Dot one Q8_0 weight row (f16 scale + 32 int8 per block) against a
 * pre-quantized Q8_0 activation (copied from ds4.c dot_q8_0_row). */
static inline float ds4gk_dot_q8_0_row(
        const uint8_t *row,
        const int8_t  *xq,
        const float   *xscale,
        uint64_t       in_dim,
        uint64_t       blocks) {
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34 + 2);

        const uint64_t i0 = b * 32;
        const uint64_t n = in_dim - i0 < 32 ? in_dim - i0 : 32;
        acc += ds4gk_f16_to_f32(scale_bits) * xscale[b] * (float)ds4gk_dot_i8_32(qs, xq + i0, n);
    }
    return acc;
}

/* Q8_K activation quantization (copied from ds4.c ds4_quantize_row_q8_K). */
static void ds4gk_quantize_row_q8_K(const float *x, ds4gk_block_q8_K *y, int64_t k) {
    const int64_t nb = k / (int64_t)DS4GK_QK_K;
    for (int64_t b = 0; b < nb; b++) {
        float max = 0.0f;
        float amax = 0.0f;
        for (int j = 0; j < (int)DS4GK_QK_K; j++) {
            const float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax;
                max = x[j];
            }
        }
        if (amax == 0.0f) {
            y[b].d = 0.0f;
            memset(y[b].qs, 0, sizeof(y[b].qs));
            memset(y[b].bsums, 0, sizeof(y[b].bsums));
            x += DS4GK_QK_K;
            continue;
        }
        const float iscale = -127.0f / max;
        for (int j = 0; j < (int)DS4GK_QK_K; j++) {
            int v = (int)lrintf(iscale * x[j]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            y[b].qs[j] = (int8_t)v;
        }
        for (int j = 0; j < (int)(DS4GK_QK_K / 16); j++) {
            int sum = 0;
            for (int i = 0; i < 16; i++) sum += y[b].qs[j * 16 + i];
            y[b].bsums[j] = (int16_t)sum;
        }
        y[b].d = 1.0f / iscale;
        x += DS4GK_QK_K;
    }
}

static inline int32_t ds4gk_dot_q2_16(const uint8_t *q2, const int8_t *q8, int shift) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 16; i++) sum += (int32_t)q8[i] * (int32_t)((q2[i] >> shift) & 3);
    return sum;
}

/* Q2_K x Q8_K dot (plain-C path of ds4.c ds4_vec_dot_q2_K_q8_K). */
static float ds4gk_vec_dot_q2_K_q8_K(int n, const ds4gk_block_q2_K *x, const ds4gk_block_q8_K *y) {
    const int nb = n / (int)DS4GK_QK_K;
    float sumf = 0.0f;
    for (int i = 0; i < nb; i++) {
        const uint8_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        const uint8_t *sc = x[i].scales;

        int summs = 0;
        for (int j = 0; j < 16; j++) {
            summs += y[i].bsums[j] * (sc[j] >> 4);
        }

        const float dall = y[i].d * ds4gk_f16_to_f32(x[i].d);
        const float dmin = y[i].d * ds4gk_f16_to_f32(x[i].dmin);

        int isum = 0;
        int is = 0;
        for (int k = 0; k < (int)(DS4GK_QK_K / 128); k++) {
            int shift = 0;
            for (int j = 0; j < 4; j++) {
                int d = sc[is++] & 0x0f;
                int isuml = ds4gk_dot_q2_16(q2, q8, shift);
                isum += d * isuml;

                d = sc[is++] & 0x0f;
                isuml = ds4gk_dot_q2_16(q2 + 16, q8 + 16, shift);
                isum += d * isuml;

                shift += 2;
                q8 += 32;
            }
            q2 += 32;
        }
        sumf += dall * (float)isum - dmin * (float)summs;
    }
    return sumf;
}

static inline int32_t ds4gk_dot_iq2_pair_16(const int8_t *grid0, const int8_t *grid1, const int8_t *q8) {
    int32_t sum = 0;
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid0[i] * (int32_t)q8[i];
    for (uint32_t i = 0; i < 8; i++) sum += (int32_t)grid1[i] * (int32_t)q8[8 + i];
    return sum;
}

/* IQ2_XXS x Q8_K dot (plain-C path of ds4.c ds4_vec_dot_iq2_xxs_q8_K). */
static float ds4gk_vec_dot_iq2_xxs_q8_K(int n, const ds4gk_block_iq2_xxs *x, const ds4gk_block_q8_K *y) {
    ds4gk_iq2xxs_ensure();
    const int nb = n / (int)DS4GK_QK_K;
    uint32_t aux32[2];
    const uint8_t *aux8 = (const uint8_t *)aux32;
    float sumf = 0.0f;

    for (int i = 0; i < nb; i++) {
        const float d = ds4gk_f16_to_f32(x[i].d) * y[i].d;
        const uint16_t *q2 = x[i].qs;
        const int8_t *q8 = y[i].qs;
        int32_t bsum = 0;

        for (int ib32 = 0; ib32 < (int)(DS4GK_QK_K / 32); ib32++) {
            memcpy(aux32, q2, 2 * sizeof(uint32_t));
            q2 += 4;

            const uint32_t ls = 2 * (aux32[1] >> 28) + 1;
            int32_t sumi = 0;
            for (int l = 0; l < 4; l += 2) {
                const uint32_t sign_idx0 = (aux32[1] >> (7 * l)) & 127;
                const uint32_t sign_idx1 = (aux32[1] >> (7 * (l + 1))) & 127;
                sumi += ds4gk_dot_iq2_pair_16(ds4gk_iq2xxs_signed_grid[aux8[l]][sign_idx0],
                                              ds4gk_iq2xxs_signed_grid[aux8[l + 1]][sign_idx1],
                                              q8);
                q8 += 16;
            }
            bsum += sumi * (int32_t)ls;
        }
        sumf += d * (float)bsum;
    }
    return 0.125f * sumf;
}

/* One token slot of the routed MoE FFN.  Shared by the single-token API and
 * the prefill batch API (the batch is just the per-token loop over this with
 * slot offsets).  All output pointers must already be advanced to this
 * token's slot; the caller validates the tensors and dimensions.  Returns
 * false and prints a diagnostic on an unsupported quant type, an out-of-range
 * selected expert, or weights past the end of the model map. */
#if 0
static bool ds4gk_routed_moe_slot(
        uint32_t gate_type, uint32_t down_type,
        uint32_t in_dim, uint32_t mid_dim, uint32_t out_dim,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const uint8_t *base, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        const float *xp, const int32_t *selp, const float *wp,
        float *gp, float *upp, float *mp, float *ep, float *op,
        const char *tag)
{
    /* Range helper: rows*row_bytes must fit inside [off, model_size). */
    auto range_ok = [model_size](uint64_t off, uint64_t rows, uint64_t row_bytes) -> bool {
        if (row_bytes == 0 || off > model_size) return false;
        return rows <= (model_size - off) / row_bytes;
    };

    const bool gate_q8_0 = (gate_type == 8);
    const bool gate_q2_k = (gate_type == 10);
    const bool gate_iq2  = (gate_type == 16);

    /* Input quantization: Q8_0 rows need a Q8_0 activation, Q2_K / IQ2_XXS
     * rows need a Q8_K activation (mirrors ds4.c matvec_*_prequant). */
    std::vector<int8_t> xq8;
    std::vector<float>  xscale8;
    std::vector<ds4gk_block_q8_K> xqk;
    if (gate_q8_0) {
        const uint64_t blocks = (in_dim + 31) / 32;
        xq8.resize(blocks * 32);
        xscale8.resize(blocks);
        ds4gk_quantize_q8_0_activation(xp, xq8.data(), xscale8.data(), in_dim);
    } else {
        if (!gate_q2_k && !gate_iq2) {
            fprintf(stderr, "ds4: %s: unsupported gate type %u\n", tag, gate_type);
            return false;
        }
        if (in_dim % DS4GK_QK_K != 0) {
            fprintf(stderr, "ds4: %s: QK_K-aligned expert input required for type %u\n", tag, gate_type);
            return false;
        }
        xqk.resize(in_dim / DS4GK_QK_K);
        ds4gk_quantize_row_q8_K(xp, xqk.data(), in_dim);
    }

    /* Per-expert mid (weighted SwiGLU), re-quantized before the down dot. */
    std::vector<float> midv(mid_dim);
    std::vector<int8_t> midq8;
    std::vector<float> midscale8;
    std::vector<ds4gk_block_q8_K> midqk;
    const uint64_t mid_q8_0_blocks = (mid_dim + 31) / 32;
    const uint64_t mid_q8k_blocks  = mid_dim / DS4GK_QK_K;
    const bool down_q8_0 = (down_type == 8);
    const bool down_q2_k = (down_type == 10);
    const bool down_iq2  = (down_type == 16);
    if (down_q8_0) {
        midq8.resize(mid_q8_0_blocks * 32);
        midscale8.resize(mid_q8_0_blocks);
    } else {
        if (!down_q2_k && !down_iq2) {
            fprintf(stderr, "ds4: %s: unsupported down type %u\n", tag, down_type);
            return false;
        }
        if (mid_dim % DS4GK_QK_K != 0) {
            fprintf(stderr, "ds4: %s: QK_K-aligned expert mid required for down type %u\n", tag, down_type);
            return false;
        }
        midqk.resize(mid_q8k_blocks);
    }

    memset(op, 0, (uint64_t)out_dim * sizeof(float));

    for (uint32_t e = 0; e < n_expert; e++) {
        const int32_t expert = selp[e];
        if (expert < 0 || (uint32_t)expert >= n_total_expert) {
            fprintf(stderr, "ds4: %s: selected expert %d out of range [0,%u)\n",
                    tag, expert, n_total_expert);
            return false;
        }
        const float weight = wp[e];

        const uint64_t gate_ex_off = gate_offset + (uint64_t)expert * gate_expert_bytes;
        const uint64_t up_ex_off   = up_offset   + (uint64_t)expert * gate_expert_bytes;
        const uint64_t down_ex_off = down_offset + (uint64_t)expert * down_expert_bytes;
        if (!range_ok(gate_ex_off, mid_dim, gate_row_bytes) ||
            !range_ok(up_ex_off, mid_dim, gate_row_bytes) ||
            !range_ok(down_ex_off, out_dim, down_row_bytes)) {
            fprintf(stderr, "ds4: %s: expert %d weights out of model map range\n", tag, expert);
            return false;
        }
        const uint8_t *gate_base = base + gate_ex_off;
        const uint8_t *up_base   = base + up_ex_off;
        const uint8_t *down_base = base + down_ex_off;

        /* gate/up projection, clamp, SwiGLU, router weight (ds4.c order). */
        for (uint32_t r = 0; r < mid_dim; r++) {
            float gval = 0.0f;
            float uval = 0.0f;
            if (gate_q8_0) {
                gval = ds4gk_dot_q8_0_row(gate_base + (uint64_t)r * gate_row_bytes,
                                          xq8.data(), xscale8.data(),
                                          in_dim, xq8.size() / 32);
                uval = ds4gk_dot_q8_0_row(up_base + (uint64_t)r * gate_row_bytes,
                                          xq8.data(), xscale8.data(),
                                          in_dim, xq8.size() / 32);
            } else if (gate_q2_k) {
                gval = ds4gk_vec_dot_q2_K_q8_K(in_dim,
                                               (const ds4gk_block_q2_K *)(gate_base + (uint64_t)r * gate_row_bytes),
                                               xqk.data());
                uval = ds4gk_vec_dot_q2_K_q8_K(in_dim,
                                               (const ds4gk_block_q2_K *)(up_base + (uint64_t)r * gate_row_bytes),
                                               xqk.data());
            } else { /* gate_iq2 */
                gval = ds4gk_vec_dot_iq2_xxs_q8_K(in_dim,
                                                  (const ds4gk_block_iq2_xxs *)(gate_base + (uint64_t)r * gate_row_bytes),
                                                  xqk.data());
                uval = ds4gk_vec_dot_iq2_xxs_q8_K(in_dim,
                                                  (const ds4gk_block_iq2_xxs *)(up_base + (uint64_t)r * gate_row_bytes),
                                                  xqk.data());
            }

            if (clamp > 1.0e-6f) {
                if (gval > clamp) gval = clamp;
                if (uval > clamp) uval = clamp;
                if (uval < -clamp) uval = -clamp;
            }
            const float mval = ds4gk_silu(gval) * uval * weight;
            gp[(uint64_t)e * mid_dim + r] = gval;
            upp[(uint64_t)e * mid_dim + r] = uval;
            mp[(uint64_t)e * mid_dim + r] = mval;
            midv[r] = mval;
        }

        /* Down projection on the re-quantized weighted mid. */
        if (down_q8_0) {
            ds4gk_quantize_q8_0_activation(midv.data(), midq8.data(), midscale8.data(),
                                           mid_dim);
            for (uint32_t r2 = 0; r2 < out_dim; r2++) {
                const float dv = ds4gk_dot_q8_0_row(down_base + (uint64_t)r2 * down_row_bytes,
                                                    midq8.data(), midscale8.data(),
                                                    mid_dim, mid_q8_0_blocks);
                ep[(uint64_t)e * out_dim + r2] = dv;
                op[r2] += dv;
            }
        } else {
            ds4gk_quantize_row_q8_K(midv.data(), midqk.data(), mid_dim);
            if (down_q2_k) {
                for (uint32_t r2 = 0; r2 < out_dim; r2++) {
                    const float dv = ds4gk_vec_dot_q2_K_q8_K(mid_dim,
                                                             (const ds4gk_block_q2_K *)(down_base + (uint64_t)r2 * down_row_bytes),
                                                             midqk.data());
                    ep[(uint64_t)e * out_dim + r2] = dv;
                    op[r2] += dv;
                }
            } else { /* down_iq2 */
                for (uint32_t r2 = 0; r2 < out_dim; r2++) {
                    const float dv = ds4gk_vec_dot_iq2_xxs_q8_K(mid_dim,
                                                                (const ds4gk_block_iq2_xxs *)(down_base + (uint64_t)r2 * down_row_bytes),
                                                                midqk.data());
                    ep[(uint64_t)e * out_dim + r2] = dv;
                    op[r2] += dv;
                }
            }
        }
    }
    return true;
}
#endif

struct ds4gk_routed_pc {
    uint32_t mode, gate_type, down_type, in_dim, mid_dim, out_dim;
    uint32_t n_tokens, n_selected, n_total_expert;
    uint32_t gate_expert_bytes, gate_row_bytes;
    uint32_t down_expert_bytes, down_row_bytes, q8_blocks;
    float clamp_value;
    uint32_t add_enabled, q2_words;
};

static bool ds4gk_routed_buffer(const ds4_gpu_tensor *tensor,
                                 VkDescriptorBufferInfo &info) {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize offset = 0;
    if (!find_tensor_buffer(tensor, buffer, offset) || tensor->bytes == 0)
        return false;
    const VkDeviceSize alignment = (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if ((alignment != 0 && offset % alignment != 0) ||
        tensor->bytes > g_vk.caps.max_storage_buffer_range) return false;
    /* tensor->bytes is the view length; offset belongs to the parent buffer. */
    info = {buffer, offset, tensor->bytes};
    return info.range != 0;
}

static bool ds4gk_routed_model(uint64_t offset, uint64_t bytes,
                               VkDescriptorBufferInfo &info) {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceSize buffer_offset = 0, range = 0;
    if (!find_model_buffer(offset, bytes, buffer, buffer_offset, range)) return false;
    const VkDeviceSize alignment = (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment;
    if ((alignment != 0 && buffer_offset % alignment != 0) ||
        range > g_vk.caps.max_storage_buffer_range) return false;
    info = {buffer, buffer_offset, range};
    return info.range != 0;
}

static const char *ds4gk_routed_mode_shader(const char *fallback,
                                             uint32_t mode) {
    const char *enabled = getenv("DS4_VULKAN_ROUTED_SPECIALIZED");
    if (enabled && (*enabled == '0' || *enabled == 'n' || *enabled == 'N'))
        return fallback;
    if (mode > 5u) return fallback;
    static const char *const names[] = {
        "routed_moe_mode0", "routed_moe_mode1", "routed_moe_mode2",
        "routed_moe_mode3", "routed_moe_mode4", "routed_moe_mode5"};
    const char *name = names[mode];
    return g_vk.shader_map.find(name) != g_vk.shader_map.end() ? name : fallback;
}

/* The Wave64 routed variants are the BC-250 appliance default. They are only
 * valid for the 16-lane/8-lane groups inside a reported 64-lane subgroup;
 * unusual devices retain the ordinary shaders, and =0 remains an exact
 * same-binary fallback for diagnosis. */
static bool ds4gk_routed_wave64_enabled(void) {
    const char *enabled = getenv("DS4_VULKAN_ROUTED_WAVE64");
    if (enabled && (*enabled == '0' || *enabled == 'n' || *enabled == 'N'))
        return false;
    return g_vk.caps.subgroup_size == 64u && g_vk.caps.has_subgroup_shuffle;
}

static const char *ds4gk_routed_shape_shader(const char *ordinary,
                                              const char *wave64) {
    return ds4gk_routed_wave64_enabled() &&
        g_vk.shader_map.find(wave64) != g_vk.shader_map.end() ? wave64 : ordinary;
}

/* The production Flash decode appliance already keeps router IDs and route
 * weights on the device for the whole layer command batch. On that narrow
 * shape a compact fused shader writes Q8 mid directly; gate/up/f32-mid
 * tensors and their descriptor bindings do not exist. */
static bool ds4gk_routed_mid_only_appliance(
        uint32_t gate_type, uint32_t down_type,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        uint32_t n_total_expert, uint32_t n_expert, uint32_t n_tokens) {
    const char *enabled = getenv("DS4_VULKAN_ROUTED_MID_ONLY");
    if (enabled && (*enabled == '0' || *enabled == 'n' || *enabled == 'N'))
        return false;
    const char *iq2_words = getenv("DS4_VULKAN_ROUTED_IQ2_WORDS");
    if (iq2_words && strcmp(iq2_words, "0") == 0) return false;
    return g_vk.shader_map.find("routed_moe_fused_mid") != g_vk.shader_map.end() &&
        get_cmd_ctx().layer_batch_active && n_tokens == 1u &&
        gate_type == 16u && down_type == 10u &&
        expert_in_dim == 4096u && expert_mid_dim == 2048u &&
        out_dim == 4096u && n_total_expert == 256u && n_expert == 6u;
}

/* Q2 down has eight q8 blocks for Flash (2048 intermediate values).  The
 * fused appliance assigns six 8-lane slots to each output row, so it can
 * perform the old down reduction and rank-ascending weighted sum in one
 * workgroup without atomics or a changed FP32 accumulation order. */
static bool ds4gk_routed_down_reduce_appliance(
        uint32_t down_type, uint32_t expert_in_dim,
        uint32_t expert_mid_dim, uint32_t out_dim,
        uint32_t n_total_expert, uint32_t n_expert, uint32_t n_tokens) {
    const char *enabled = getenv("DS4_VULKAN_ROUTED_DOWN_REDUCE");
    if (enabled && (*enabled == '0' || *enabled == 'n' || *enabled == 'N'))
        return false;
    return g_vk.shader_map.find("routed_moe_down_reduce_q2") != g_vk.shader_map.end() &&
        get_cmd_ctx().layer_batch_active && n_tokens == 1u &&
        down_type == 10u && expert_in_dim == 4096u &&
        expert_mid_dim == 2048u && out_dim == 4096u &&
        n_total_expert == 256u && n_expert == 6u;
}

/* Canonical routed stages use a fixed descriptor ABI: b4 is the stage output,
 * while the generic fused gate/up shader additionally writes b5 and b6. The
 * compact mid-only shader writes b4; the compact Q2 appliance writes b3
 * because it does not bind the dead expert-output scratch.
 * Keep dependencies scoped to actual output ranges so routed dispatches do
 * not publish unrelated allocations through a global memory barrier. */
static void ds4gk_routed_output_barrier(
        VulkanCommandCtx &ctx, const char *shader_name,
        VkDescriptorBufferInfo *buffers, uint32_t buffer_count) {
    VkBufferMemoryBarrier barriers[3] = {};
    uint32_t count = 0;
    auto add = [&](uint32_t binding) {
        if (binding >= buffer_count || buffers[binding].buffer == VK_NULL_HANDLE ||
            buffers[binding].range == 0 || count >= 3) return;
        VkBufferMemoryBarrier &barrier = barriers[count++];
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT |
                                VK_ACCESS_SHADER_WRITE_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffers[binding].buffer;
        barrier.offset = buffers[binding].offset;
        barrier.size = buffers[binding].range;
    };
    const bool down_reduce = shader_name &&
        (strcmp(shader_name, "routed_moe_down_reduce_q2") == 0 ||
         strcmp(shader_name, "routed_moe_down_reduce_q2_wave64") == 0);
    add(down_reduce
            ? 3u : 4u);
    if (shader_name && strcmp(shader_name, "routed_moe_fused") == 0) {
        add(5);
        add(6);
    }
    if (count == 0) return;
    timeline_barrier(ctx, "routed_moe_output_dependency");
    vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr,
                         count, barriers, 0, nullptr);
}

/* Publish router-selected IDs/weights only when the routed consumer needs
 * them. This leaves independent shared-expert work between router selection
 * and the first routed consumer in a layer batch. */
static void ds4gk_routed_input_barrier(
        VulkanCommandCtx &ctx, const char *shader_name,
        const ds4gk_routed_pc &pc, VkDescriptorBufferInfo *buffers,
        uint32_t buffer_count) {
    VkBufferMemoryBarrier barriers[2] = {};
    uint32_t count = 0;
    auto add = [&](uint32_t binding) {
        if (binding >= buffer_count || buffers[binding].buffer == VK_NULL_HANDLE ||
            buffers[binding].range == 0 || count >= 2) return;
        VkBufferMemoryBarrier &barrier = barriers[count++];
        barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        barrier.buffer = buffers[binding].buffer;
        barrier.offset = buffers[binding].offset;
        barrier.size = buffers[binding].range;
    };
    const bool down_reduce = shader_name &&
        (strcmp(shader_name, "routed_moe_down_reduce_q2") == 0 ||
         strcmp(shader_name, "routed_moe_down_reduce_q2_wave64") == 0);
    if (pc.mode == 1u || pc.mode == 3u) {
        add(down_reduce
                ? 2u : 3u); /* selected IDs */
    }
    if (pc.mode == 2u) add(2);                  /* router weights */
    if (shader_name && strcmp(shader_name, "routed_moe_fused") == 0)
        add(7);                                 /* fused weights */
    if (shader_name && (strcmp(shader_name, "routed_moe_fused_mid") == 0 ||
                        strcmp(shader_name, "routed_moe_fused_mid_wave64") == 0))
        add(5);                                 /* compact fused weights */
    if (count == 0) return;
    timeline_barrier(ctx, "routed_moe_input_dependency");
    vkCmdPipelineBarrier(ctx.cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT |
                             VK_PIPELINE_STAGE_HOST_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 0, nullptr, count, barriers, 0, nullptr);
}

static bool ds4gk_routed_dispatch_shader(
        const char *shader_name, const char *stage,
        const ds4gk_routed_pc &pc, VkDescriptorBufferInfo *buffers,
        uint32_t buffer_count, uint32_t gx, uint32_t gy, uint32_t gz,
        std::vector<VkDescriptorSet> &sets) {
    auto si = g_vk.shader_map.find(shader_name);
    if (si == g_vk.shader_map.end()) return false;
    auto &ctx = get_cmd_ctx();
    if (!ctx.recording && !begin_cmd()) return false;
    auto &shader = g_vk.shaders[si->second];
    VkDescriptorSet set = VK_NULL_HANDLE;
    if (!allocate_simple_descriptors(shader, buffers, buffer_count, set)) return false;
    vkCmdBindPipeline(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, shader.pipeline);
    vkCmdBindDescriptorSets(ctx.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            shader.layout, 0, 1, &set, 0, nullptr);
    vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(pc), &pc);
    ds4gk_routed_input_barrier(ctx, shader_name, pc, buffers, buffer_count);
    uint32_t first_query = UINT32_MAX;
    if (ctx.routed_profile_enabled && ctx.timestamp_pool != VK_NULL_HANDLE &&
        ctx.timestamp_cursor + 1u < DS4_VK_TIMELINE_QUERY_COUNT) {
        first_query = ctx.timestamp_cursor;
        ctx.timestamp_cursor += 2;
        vkCmdWriteTimestamp(ctx.cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                            ctx.timestamp_pool, first_query);
    }
    timeline_dispatch(ctx, stage, buffers, buffer_count, gx, gy, gz);
    ctx.command_count++;
    ds4gk_routed_output_barrier(ctx, shader_name, buffers, buffer_count);
    if (first_query != UINT32_MAX) {
        vkCmdWriteTimestamp(ctx.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                            ctx.timestamp_pool, first_query + 1);
        ctx.routed_timestamps.push_back({stage, first_query});
        /* Profiling mode serializes query boundaries so adjacent dispatch
         * intervals cannot overlap or double-count GPU time. */
        timeline_barrier(ctx, "routed_moe_profile_serialization");
        vkCmdPipelineBarrier(ctx.cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, 0, nullptr,
                             0, nullptr, 0, nullptr);
    }
    sets.push_back(set);
    return true;
}

static bool ds4gk_routed_dispatch(const char *stage,
                                  const ds4gk_routed_pc &pc,
                                  VkDescriptorBufferInfo *buffers,
                                  uint32_t gx, uint32_t gy, uint32_t gz,
                                  std::vector<VkDescriptorSet> &sets) {
    return ds4gk_routed_dispatch_shader(
        ds4gk_routed_mode_shader("routed_moe", pc.mode),
        stage, pc, buffers, 6, gx, gy, gz, sets);
}

static uint32_t ds4gk_routed_projection_groups(uint32_t rows,
                                               uint32_t blocks) {
    uint32_t lanes = 8;
    while (lanes < blocks && lanes < 256u) lanes <<= 1u;
    const uint32_t rows_per_group = 256u / lanes;
    return (rows + rows_per_group - 1u) / rows_per_group;
}

static bool ds4gk_routed_flush(std::vector<VkDescriptorSet> &sets) {
    if (sets.empty()) return true;
    const uint64_t flush_start = timeline_now_ns();
    bool ok = submit_and_wait() != 0;
    if (getenv("DS4_VULKAN_TIMELINE"))
        timeline_duration_current(TimelineEventKind::Wait,
                                  "routed_tile_flush", flush_start);
    for (VkDescriptorSet set : sets)
        if (!release_simple_descriptors(set)) ok = false;
    sets.clear();
    return ok;
}

static bool ds4gk_routed_common(
        ds4_gpu_tensor *out, ds4_gpu_tensor *gate, ds4_gpu_tensor *up,
        ds4_gpu_tensor *mid, ds4_gpu_tensor *experts,
        const void *model_map, uint64_t model_size,
        uint64_t gate_offset, uint64_t up_offset, uint64_t down_offset,
        uint32_t gate_type, uint32_t down_type,
        uint64_t gate_expert_bytes, uint64_t gate_row_bytes,
        uint64_t down_expert_bytes, uint64_t down_row_bytes,
        uint32_t expert_in_dim, uint32_t expert_mid_dim, uint32_t out_dim,
        const ds4_gpu_tensor *selected, const ds4_gpu_tensor *weights,
        uint32_t n_total_expert, uint32_t n_expert, float clamp,
        const ds4_gpu_tensor *x, const ds4_gpu_tensor *add_in,
        uint32_t n_tokens, bool *mid_is_f16, bool validate_selected) {
    /* A layer command batch consumes GPU-produced IDs; the routed shaders
     * already bounds-check them, so its one-token and batched calls pass
     * validate_selected=false.  Calls outside a batch retain validation for
     * focused CPU-written invalid-ID tests. */
    const bool fused_down_reduce = ds4gk_routed_down_reduce_appliance(
        down_type, expert_in_dim, expert_mid_dim, out_dim,
        n_total_expert, n_expert, n_tokens);
    const bool mid_only = ds4gk_routed_mid_only_appliance(
        gate_type, down_type, expert_in_dim, expert_mid_dim, out_dim,
        n_total_expert, n_expert, n_tokens);
    if (mid_is_f16) *mid_is_f16 = false;
    if (!out || !mid || !selected || !weights ||
        !x || !model_map || n_tokens == 0 || n_expert == 0 || n_total_expert == 0 ||
        expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        (gate_type != 8 && gate_type != 10 && gate_type != 16) ||
        (down_type != 8 && down_type != 10 && down_type != 16) ||
        (gate_type != 8 && expert_in_dim % 256 != 0) ||
        (down_type != 8 && expert_mid_dim % 256 != 0))
        return false;
    if (g_vk.model_map != model_map || g_vk.model_size != model_size)
        set_model_map_identity(model_map, model_size);
    const uint64_t gate_blocks = gate_type == 8 ? (expert_in_dim + 31u) / 32u : expert_in_dim / 256;
    const uint64_t mid_blocks = down_type == 8 ? (expert_mid_dim + 31u) / 32u : expert_mid_dim / 256;
    const uint64_t expected_gate_row = gate_blocks *
        (gate_type == 8 ? 34 : gate_type == 10 ? 84 : 66);
    const uint64_t expected_down_row = mid_blocks * (down_type == 8 ? 34 : down_type == 10 ? 84 : 66);
    if (gate_row_bytes != expected_gate_row || down_row_bytes != expected_down_row ||
        gate_expert_bytes != (uint64_t)expert_mid_dim * gate_row_bytes ||
        down_expert_bytes != (uint64_t)out_dim * down_row_bytes ||
        gate_expert_bytes > UINT32_MAX || gate_row_bytes > UINT32_MAX ||
        down_expert_bytes > UINT32_MAX || down_row_bytes > UINT32_MAX ||
        n_tokens > UINT32_MAX || expert_in_dim > UINT32_MAX ||
        expert_mid_dim > UINT32_MAX || out_dim > UINT32_MAX)
        return false;
    uint64_t gate_bytes, down_bytes;
    if (!checked_u64_product(n_total_expert, gate_expert_bytes, gate_bytes) ||
        !checked_u64_product(n_total_expert, down_expert_bytes, down_bytes) ||
        gate_offset > model_size || gate_bytes > model_size - gate_offset ||
        up_offset > model_size || gate_bytes > model_size - up_offset ||
        down_offset > model_size || down_bytes > model_size - down_offset)
        return false;
    uint64_t selected_values, pair_values, expert_values, x_values, out_values;
    if (!checked_u64_product(n_tokens, n_expert, selected_values) ||
        !checked_u64_product(selected_values, expert_mid_dim, pair_values) ||
        !checked_u64_product(selected_values, out_dim, expert_values) ||
        !checked_u64_product(n_tokens, expert_in_dim, x_values) ||
        !checked_u64_product(n_tokens, out_dim, out_values)) return false;
    if (selected_values > UINT32_MAX || pair_values > UINT32_MAX ||
        expert_values > UINT32_MAX ||
        selected->bytes < selected_values * sizeof(int32_t) ||
        weights->bytes < selected_values * sizeof(float) ||
        x->bytes < x_values * sizeof(float) ||
        (gate && gate->bytes < pair_values * sizeof(float)) ||
        (up && up->bytes < pair_values * sizeof(float)) ||
        (experts && experts->bytes < expert_values * sizeof(float)) ||
        out->bytes < out_values * sizeof(float) ||
        (add_in && add_in->bytes < (uint64_t)out_dim * sizeof(float)))
        return false;

    /* The decode appliance does not expose gate/up or per-expert down
     * intermediates. Keep the public API fallback intact by allocating those
     * tensors only when a diagnostic switch or unsupported shape selects the
     * generic stages. The destructor also covers every early-return path. */
    struct RoutedOwnedScratch {
        ds4_gpu_tensor gate{}, up{}, mid{}, experts{};
        ~RoutedOwnedScratch() {
            ds4_gpu_tensor_free_in_place(&experts);
            ds4_gpu_tensor_free_in_place(&mid);
            ds4_gpu_tensor_free_in_place(&up);
            ds4_gpu_tensor_free_in_place(&gate);
        }
    } owned;
    const uint64_t pair_bytes = pair_values * sizeof(float);
    const uint64_t expert_bytes = expert_values * sizeof(float);
    if (!gate && !mid_only) {
        if (ds4_gpu_tensor_alloc_on(&owned.gate, 0, pair_bytes) != 0)
            return false;
        gate = &owned.gate;
    }
    if (!up && !mid_only) {
        if (ds4_gpu_tensor_alloc_on(&owned.up, 0, pair_bytes) != 0)
            return false;
        up = &owned.up;
    }
    if (!mid_only && mid->bytes < pair_bytes) {
        if (ds4_gpu_tensor_alloc_on(&owned.mid, 0, pair_bytes) != 0)
            return false;
        mid = &owned.mid;
    }
    if (!experts && !fused_down_reduce) {
        if (ds4_gpu_tensor_alloc_on(&owned.experts, 0, expert_bytes) != 0)
            return false;
        experts = &owned.experts;
    }

    VkDescriptorBufferInfo x_info, out_info, gate_info, up_info, mid_info, exp_info;
    VkDescriptorBufferInfo gate_model, up_model, down_model;
    VkDescriptorBufferInfo selected_info, weights_info, add_info;
    if (!ds4gk_routed_buffer(x, x_info) || !ds4gk_routed_buffer(out, out_info) ||
        (!mid_only && (!ds4gk_routed_buffer(gate, gate_info) ||
                       !ds4gk_routed_buffer(up, up_info))) ||
        !ds4gk_routed_buffer(mid, mid_info) ||
        (!fused_down_reduce && !ds4gk_routed_buffer(experts, exp_info)) ||
        !ds4gk_routed_buffer(selected, selected_info) || !ds4gk_routed_buffer(weights, weights_info))
        return false;
    if (add_in && !ds4gk_routed_buffer(add_in, add_info)) return false;
    if (!add_in) add_info = out_info;
    if (!ds4gk_routed_model(gate_offset, gate_bytes, gate_model) ||
        !ds4gk_routed_model(up_offset, gate_bytes, up_model) ||
        !ds4gk_routed_model(down_offset, down_bytes, down_model))
        return false;

    uint64_t q8_blocks = 0, q8_bytes = 0;
    uint64_t input_q8_bytes = 0, mid_q8_bytes = 0;
    const uint64_t q8_stride = gate_type == 8 ? 36 : 292;
    if (!checked_u64_product((uint64_t)n_tokens * n_expert,
                             std::max(gate_blocks, mid_blocks), q8_blocks) ||
        !checked_u64_product(q8_blocks,
                             std::max<uint64_t>(q8_stride, down_type == 8 ? 36 : 292), q8_bytes) ||
        !checked_u64_product((uint64_t)n_tokens * gate_blocks,
                             q8_stride, input_q8_bytes) ||
        !checked_u64_product((uint64_t)n_tokens * n_expert * mid_blocks,
                             down_type == 8 ? 36u : 292u, mid_q8_bytes) ||
        q8_bytes > UINT32_MAX * (uint64_t)sizeof(uint32_t))
        return false;
    if (gate_blocks > g_vk.caps.max_compute_work_group_count[0] ||
        n_tokens > g_vk.caps.max_compute_work_group_count[1] ||
        n_expert > g_vk.caps.max_compute_work_group_count[2] ||
        expert_mid_dim > g_vk.caps.max_compute_work_group_count[0] ||
        out_dim > g_vk.caps.max_compute_work_group_count[0] ||
        g_vk.caps.max_compute_work_group_size[0] < 256 ||
        g_vk.caps.max_compute_work_group_invocations < 256)
        return false;
    ds4_gpu_tensor q8{}, compact_mid_q8{}, invalid{};
    const VkDeviceSize storage_alignment =
        std::max<VkDeviceSize>(1u,
            (VkDeviceSize)g_vk.caps.min_storage_buffer_offset_alignment);
    /* The compact appliance owns each complete 256-value Q8_K block in one
     * workgroup. It serializes the same 16 projection row tiles inside that
     * group, preserving their reduction order and the canonical max/rounding
     * rules while avoiding a device-wide rendezvous and f32 materialization. */
    const uint64_t mid_q8_offset = mid_only
        ? (input_q8_bytes + storage_alignment - 1u) /
              storage_alignment * storage_alignment
        : 0u;
    const bool persistent_q8 = mid_only && mid_q8_offset <= mid->bytes &&
        mid_q8_bytes <= mid->bytes - mid_q8_offset;
    if (persistent_q8) {
        q8.ptr = mid->ptr;
        q8.bytes = input_q8_bytes;
        q8.owner = 0;
        q8.device_id = mid->device_id;
        compact_mid_q8.ptr = (char *)mid->ptr + mid_q8_offset;
        compact_mid_q8.bytes = mid_q8_bytes;
        compact_mid_q8.owner = 0;
        compact_mid_q8.device_id = mid->device_id;
    }
    if ((!persistent_q8 &&
         ds4_gpu_tensor_alloc_host_scratch_in_place(&q8, q8_bytes) != 0) ||
        (validate_selected &&
         ds4_gpu_tensor_alloc_host_scratch_in_place(&invalid,
                                                    sizeof(uint32_t)) != 0)) {
        ds4_gpu_tensor_free_in_place(&q8);
        ds4_gpu_tensor_free_in_place(&invalid);
        return false;
    }
    const uint32_t zero = 0;
    if (validate_selected &&
        !ds4_gpu_tensor_write(&invalid, 0, &zero, sizeof(zero))) {
        ds4_gpu_tensor_free_in_place(&invalid);
        ds4_gpu_tensor_free_in_place(&q8);
        return false;
    }
    const bool resume_recording = get_cmd_ctx().recording;
    bool ok = true;
    std::vector<VkDescriptorSet> sets;
    ds4gk_routed_pc pc{};
    pc = {0, gate_type, down_type, expert_in_dim, expert_mid_dim, out_dim,
          n_tokens, n_expert, n_total_expert, (uint32_t)gate_expert_bytes,
          (uint32_t)gate_row_bytes, (uint32_t)down_expert_bytes,
             (uint32_t)down_row_bytes, (uint32_t)gate_blocks, clamp, 0, 0};
    VkDescriptorBufferInfo q8_info, compact_mid_q8_info{}, invalid_info{}, iq2_lut_info;
    ok = ds4gk_routed_iq2_lut_ensure() &&
         ds4gk_routed_buffer(&q8, q8_info) &&
         (!mid_only || (persistent_q8 &&
                        ds4gk_routed_buffer(&compact_mid_q8,
                                            compact_mid_q8_info))) &&
         (!validate_selected || ds4gk_routed_buffer(&invalid, invalid_info)) &&
         ds4gk_routed_buffer(&g_vk.routed_iq2_lut, iq2_lut_info);
    if (ok && validate_selected) {
        pc.mode = 5;
        VkDescriptorBufferInfo validate_buffers[6] = {
            selected_info, selected_info, selected_info, selected_info,
            invalid_info, iq2_lut_info};
        ok = ds4gk_routed_dispatch("validate_selected", pc,
            validate_buffers, 1, n_tokens, 1, sets) &&
             ds4gk_routed_flush(sets);
        uint32_t invalid_value = 0;
        if (ok) ok = ds4_gpu_tensor_read(&invalid, 0, &invalid_value,
                                          sizeof(invalid_value)) != 0 &&
                     invalid_value == 0;
        pc.mode = 0;
    }
    if (ok) {
        VkDescriptorBufferInfo quantize_buffers[6] = {
            x_info, x_info, x_info, x_info, q8_info, iq2_lut_info};
        ok = ds4gk_routed_dispatch("quantize_input", pc,
            quantize_buffers, gate_blocks, n_tokens, 1, sets);
        if (ok && getenv("DS4_VULKAN_DEBUG")) {
            ok = ds4gk_routed_flush(sets);
            uint32_t words[3] = {};
            float scale = 0.0f;
            if (ds4_gpu_tensor_read(&q8, 0, words, sizeof(words))) {
                memcpy(&scale, &words[0], sizeof(scale));
                fprintf(stderr, "ds4: [dbg] routed input_q d=%g q=%08x/%08x\n",
                        (double)scale, words[1], words[2]);
            }
        }
    }
    bool fused_gate_up = false;
    if (ok) {
        pc.mode = 1;
        const char *iq2_words_env = getenv("DS4_VULKAN_ROUTED_IQ2_WORDS");
        /* Unaligned four-byte reconstruction may fetch the following uint.
         * Keep the exact legacy path for a descriptor with a partial tail. */
        const bool iq2_words = gate_type == 16 && (gate_bytes & 3u) == 0u &&
            (!iq2_words_env || strcmp(iq2_words_env, "0") != 0);
        pc.q2_words = iq2_words ? 1u : 0u;
        fused_gate_up = iq2_words && (mid_only ||
            g_vk.shader_map.find("routed_moe_fused") != g_vk.shader_map.end());
        if (mid_only) {
            VkDescriptorBufferInfo fused_buffers[7] = {
                q8_info, gate_model, up_model, selected_info,
                compact_mid_q8_info,
                weights_info, iq2_lut_info};
            const char *fused_mid_shader = ds4gk_routed_shape_shader(
                "routed_moe_fused_mid", "routed_moe_fused_mid_wave64");
            ok = ds4gk_routed_dispatch_shader(
                fused_mid_shader, "gate_up_swiglu_iq2_q8", pc,
                fused_buffers, 7,
                (expert_mid_dim + 255u) / 256u,
                n_tokens, n_expert, sets);
        } else if (fused_gate_up) {
            VkDescriptorBufferInfo fused_buffers[9] = {
                q8_info, gate_model, up_model, selected_info, gate_info,
                up_info, mid_info, weights_info, iq2_lut_info};
            ok = ds4gk_routed_dispatch_shader(
                "routed_moe_fused", "gate_up_swiglu_iq2", pc,
                fused_buffers, 9,
                ds4gk_routed_projection_groups(expert_mid_dim, pc.q8_blocks),
                n_tokens, n_expert, sets);
        } else {
            VkDescriptorBufferInfo gate_buffers[6] = {
                q8_info, gate_model, gate_model, selected_info, gate_info,
                iq2_lut_info};
            ok = ds4gk_routed_dispatch(iq2_words ? "gate_iq2_words" : "gate", pc,
                gate_buffers,
                ds4gk_routed_projection_groups(expert_mid_dim, pc.q8_blocks),
                n_tokens, n_expert, sets);
            if (ok) {
                pc.add_enabled = 1;
                VkDescriptorBufferInfo up_buffers[6] = {
                    q8_info, up_model, up_model, selected_info, up_info,
                    iq2_lut_info};
                ok = ds4gk_routed_dispatch(iq2_words ? "up_iq2_words" : "up", pc,
                    up_buffers,
                    ds4gk_routed_projection_groups(expert_mid_dim, pc.q8_blocks),
                    n_tokens, n_expert, sets);
                pc.add_enabled = 0;
            }
        }
        pc.q2_words = 0;
    }
    if (ok && !fused_gate_up) {
        pc.mode = 2;
        VkDescriptorBufferInfo swiglu_buffers[6] = {
            gate_info, up_info, weights_info, weights_info, mid_info, iq2_lut_info};
        ok = ds4gk_routed_dispatch("swiglu", pc,
            swiglu_buffers,
            (expert_mid_dim + 255u) / 256u, n_tokens, n_expert, sets);
    }
    if (ok && !mid_only) {
        pc.mode = 0; pc.gate_type = down_type; pc.in_dim = expert_mid_dim;
        pc.q8_blocks = (uint32_t)mid_blocks;
        pc.n_tokens = n_tokens * n_expert;
        VkDescriptorBufferInfo mid_quantize_buffers[6] = {
            mid_info, mid_info, mid_info, mid_info, q8_info, iq2_lut_info};
        ok = ds4gk_routed_dispatch("requantization", pc,
            mid_quantize_buffers,
            mid_blocks, n_tokens * n_expert, 1, sets);
        if (ok && getenv("DS4_VULKAN_DEBUG")) {
            ok = ds4gk_routed_flush(sets);
            uint32_t words[3] = {};
            float scale = 0.0f;
            if (ds4_gpu_tensor_read(&q8, 0, words, sizeof(words))) {
                memcpy(&scale, &words[0], sizeof(scale));
                fprintf(stderr, "ds4: [dbg] routed mid_q d=%g q=%08x/%08x\n",
                        (double)scale, words[1], words[2]);
            }
        }
    }
    if (mid_only) {
        pc.gate_type = down_type;
        pc.in_dim = expert_mid_dim;
        pc.q8_blocks = (uint32_t)mid_blocks;
    }
    if (ok) {
        pc.mode = 3; pc.n_tokens = n_tokens;
        pc.q2_words = down_type == 10;
        if (fused_down_reduce) {
            pc.add_enabled = add_in ? 1u : 0u;
            VkDescriptorBufferInfo down_reduce_buffers[5] = {
                mid_only ? compact_mid_q8_info : q8_info,
                down_model, selected_info, out_info, add_info};
            const char *down_reduce_shader = ds4gk_routed_shape_shader(
                "routed_moe_down_reduce_q2",
                "routed_moe_down_reduce_q2_wave64");
            const uint32_t rows_per_group = 32u / n_expert;
            ok = ds4gk_routed_dispatch_shader(
                down_reduce_shader, "down_reduce_q2", pc,
                down_reduce_buffers, 5,
                (out_dim + rows_per_group - 1u) / rows_per_group,
                n_tokens, 1, sets);
        } else {
            VkDescriptorBufferInfo down_buffers[6] = {
                mid_only ? compact_mid_q8_info : q8_info,
                down_model, down_model, selected_info, exp_info,
                iq2_lut_info};
            const char *down_stage = pc.q2_words ? "down_words" : "down_raw";
            ok = ds4gk_routed_dispatch(down_stage, pc,
                down_buffers,
                ds4gk_routed_projection_groups(out_dim, pc.q8_blocks),
                n_tokens, n_expert, sets);
            if (ok && getenv("DS4_VULKAN_DEBUG")) {
                ok = ds4gk_routed_flush(sets);
                float value = 0.0f;
                if (ds4_gpu_tensor_read(experts, 0, &value, sizeof(value)))
                    fprintf(stderr, "ds4: [dbg] routed down[0]=%g\n", (double)value);
            }
        }
    }
    if (ok && !fused_down_reduce) {
        pc.mode = 4; pc.add_enabled = add_in ? 1u : 0u;
        VkDescriptorBufferInfo reduce_buffers[6] = {
            add_info, exp_info, out_info, out_info, out_info, iq2_lut_info};
        ok = ds4gk_routed_dispatch("reduction", pc,
            reduce_buffers,
            (out_dim + 255u) / 256u, n_tokens, 1, sets);
    }
    if (!ds4gk_routed_flush(sets)) ok = false;
    ds4_gpu_tensor_free_in_place(&invalid);
    ds4_gpu_tensor_free_in_place(&q8);
    if (resume_recording && !get_cmd_ctx().recording && !begin_cmd()) ok = false;
    return ok;
}

} /* namespace */

extern "C" int ds4_gpu_routed_moe_one_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        const ds4_gpu_tensor *add_in,
        uint32_t                layer_index,
        bool                    force_resident)
{
    DS4_VK_TRACE_KERNEL("routed_moe_one");
    (void)layer_index;
    (void)force_resident;
    bool mid_is_f16 = false;
    return ds4gk_routed_common(
        out, gate, up, mid, experts, model_map, model_size,
        gate_offset, up_offset, down_offset, gate_type, down_type,
        gate_expert_bytes, gate_row_bytes, down_expert_bytes, down_row_bytes,
        expert_in_dim, expert_mid_dim, out_dim, selected, weights,
        n_total_expert, n_expert, clamp, x, add_in, 1, &mid_is_f16,
        !get_cmd_ctx().layer_batch_active) ? 1 : 0;
}

/* Prefill routed MoE over n_tokens tokens.  Same math as the single-token
 * path, looped per token with batch-slot tensor layout:
 *   selected[t*n_expert+e], weights[t*n_expert+e]
 *   gate/up/mid[t*n_expert*expert_mid_dim + e*expert_mid_dim + r]
 *   experts[t*n_expert*out_dim + e*out_dim + r2]
 *   out[t*out_dim + r2]
 * The host writes f32 mid, so *mid_is_f16 is set to false (the engine reads
 * the mid tensor as f32 afterwards; CUDA does the same). */
extern "C" int ds4_gpu_routed_moe_batch_tensor(
        ds4_gpu_tensor       *out,
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        ds4_gpu_tensor       *experts,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                down_offset,
        uint32_t                gate_type,
        uint32_t                down_type,
        uint64_t                gate_expert_bytes,
        uint64_t                gate_row_bytes,
        uint64_t                down_expert_bytes,
        uint64_t                down_row_bytes,
        uint32_t                expert_in_dim,
        uint32_t                expert_mid_dim,
        uint32_t                out_dim,
        const ds4_gpu_tensor *selected,
        const ds4_gpu_tensor *weights,
        uint32_t                n_total_expert,
        uint32_t                n_expert,
        float                   clamp,
        const ds4_gpu_tensor *x,
        uint32_t                layer_index,
        uint32_t                n_tokens,
        bool                   *mid_is_f16,
        bool                    force_resident)
{
    (void)layer_index;
    (void)force_resident;
    if (mid_is_f16) *mid_is_f16 = false;
    if (!out || !gate || !up || !mid || !experts || !selected || !weights ||
        !x || n_tokens == 0) return 0;
    constexpr uint32_t max_tokens_per_dispatch = 256u;
    const uint64_t pair_values = (uint64_t)n_expert * expert_mid_dim;
    const uint64_t expert_values = (uint64_t)n_expert * out_dim;
    if (expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        n_expert == 0 || pair_values == 0 || expert_values == 0 ||
        n_tokens > UINT64_MAX / pair_values ||
        n_tokens > UINT64_MAX / expert_values ||
        n_tokens > UINT64_MAX / expert_in_dim ||
        n_tokens > UINT64_MAX / out_dim ||
        n_tokens > UINT64_MAX / n_expert) return 0;
    auto make_view = [](ds4_gpu_tensor &view, const ds4_gpu_tensor *base,
                        uint64_t value_offset, uint64_t value_count,
                        uint64_t element_bytes) {
        if (!base || !base->ptr || value_offset > UINT64_MAX / element_bytes ||
            value_count > UINT64_MAX / element_bytes) return false;
        const uint64_t byte_offset = value_offset * element_bytes;
        const uint64_t byte_count = value_count * element_bytes;
        if (byte_offset > base->bytes || byte_count > base->bytes - byte_offset)
            return false;
        view = {};
        view.ptr = (char *)base->ptr + byte_offset;
        view.bytes = byte_count;
        view.owner = 0;
        view.device_id = base->device_id;
        return true;
    };
    for (uint32_t token_base = 0; token_base < n_tokens; ) {
        const uint32_t tile_tokens = std::min(max_tokens_per_dispatch,
                                              n_tokens - token_base);
        ds4_gpu_tensor out_view{}, gate_view{}, up_view{}, mid_view{};
        ds4_gpu_tensor experts_view{}, selected_view{}, weights_view{}, x_view{};
        if (!make_view(out_view, out, (uint64_t)token_base * out_dim,
                       (uint64_t)tile_tokens * out_dim, sizeof(float)) ||
            !make_view(gate_view, gate, (uint64_t)token_base * pair_values,
                       (uint64_t)tile_tokens * pair_values, sizeof(float)) ||
            !make_view(up_view, up, (uint64_t)token_base * pair_values,
                       (uint64_t)tile_tokens * pair_values, sizeof(float)) ||
            !make_view(mid_view, mid, (uint64_t)token_base * pair_values,
                       (uint64_t)tile_tokens * pair_values, sizeof(float)) ||
            !make_view(experts_view, experts,
                       (uint64_t)token_base * expert_values,
                       (uint64_t)tile_tokens * expert_values, sizeof(float)) ||
            !make_view(selected_view, selected,
                       (uint64_t)token_base * n_expert,
                       (uint64_t)tile_tokens * n_expert, sizeof(int32_t)) ||
            !make_view(weights_view, weights,
                       (uint64_t)token_base * n_expert,
                       (uint64_t)tile_tokens * n_expert, sizeof(float)) ||
            !make_view(x_view, x, (uint64_t)token_base * expert_in_dim,
                       (uint64_t)tile_tokens * expert_in_dim, sizeof(float)))
            return 0;
        bool tile_mid_is_f16 = false;
        if (!ds4gk_routed_common(
                &out_view, &gate_view, &up_view, &mid_view, &experts_view,
                model_map, model_size, gate_offset, up_offset, down_offset,
                gate_type, down_type, gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes, expert_in_dim,
                expert_mid_dim, out_dim, &selected_view, &weights_view,
                n_total_expert, n_expert, clamp, &x_view, nullptr,
                tile_tokens, &tile_mid_is_f16,
                !get_cmd_ctx().layer_batch_active) || tile_mid_is_f16)
            return 0;
        token_base += tile_tokens;
    }
    return 1;
}

/* =========================================================================
 * Dense quantized matmul: ds4_gpu_matmul_quant_tensor
 *
 * out[t][o] = sum_i x[t][i] * W[o][i] for the dense row-major quantized
 * weight matrix W (DS4_TENSOR_* weight_type ids, ds4.c:2040).  This is the
 * dispatch used for the dense quantized weights (Q2_K / IQ2_XXS / Q4_K and
 * the plain F16/F32/Q8_0 forms).  The plain forms delegate to the existing
 * matmul_*_tensor entries; Q2_K and IQ2_XXS are host-side ports of the
 * ds4.c matvec helpers (ds4_vec_dot_q2_K_q8_K / ds4_vec_dot_iq2_xxs_q8_K)
 * with the activation quantized to Q8_K, exactly like the routed-MoE paths.
 * The GGUF block layouts and the dot helpers are the ds4gk_* ones shared
 * with ds4_gpu_routed_moe_*_tensor above.
 * ========================================================================= */
extern "C" int ds4_gpu_matmul_quant_tensor(
        ds4_gpu_tensor       *out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint32_t                weight_type,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        uint64_t                n_tok)
{
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    switch (weight_type) {
        case 0:  /* DS4_TENSOR_F32 */
            return ds4_gpu_matmul_f32_tensor(out, model_map, model_size,
                                             weight_offset, in_dim, out_dim, x, n_tok);
        case 1:  /* DS4_TENSOR_F16 */
            return ds4_gpu_matmul_f16_tensor(out, model_map, model_size,
                                             weight_offset, in_dim, out_dim, x, n_tok);
        case 8:  /* DS4_TENSOR_Q8_0 */
            return ds4_gpu_matmul_q8_0_tensor(out, model_map, model_size,
                                              weight_offset, in_dim, out_dim, x, n_tok);
        case 10: /* DS4_TENSOR_Q2_K */
        case 16: /* DS4_TENSOR_IQ2_XXS */
            break;
        default:
            fprintf(stderr, "ds4: matmul_quant: unsupported weight_type %u\n",
                    weight_type);
            return 0;
    }

    /* Host-side Q2_K / IQ2_XXS rows: QK_K(256)-element GGUF blocks. */
    if (in_dim % DS4GK_QK_K != 0) {
        fprintf(stderr, "ds4: matmul_quant: QK_K-aligned in_dim required for type %u\n",
                weight_type);
        return 0;
    }
    const uint64_t row_bytes = (weight_type == 10)
                                   ? sizeof(ds4gk_block_q2_K)
                                   : sizeof(ds4gk_block_iq2_xxs);
    if (weight_offset > model_size || row_bytes == 0 ||
        out_dim > (model_size - weight_offset) / row_bytes) {
        fprintf(stderr, "ds4: matmul_quant: weights out of model map range\n");
        return 0;
    }
    const uint64_t x_bytes   = n_tok * in_dim * sizeof(float);
    const uint64_t out_bytes = n_tok * out_dim * sizeof(float);
    if (ds4_gpu_tensor_bytes(x) < x_bytes ||
        ds4_gpu_tensor_bytes(out) < out_bytes) {
        fprintf(stderr, "ds4: matmul_quant: tensor bytes too small\n");
        return 0;
    }

    const uint8_t *base = (const uint8_t *)model_map + weight_offset;
    const float *xp = (const float *)x->ptr;
    float *op = (float *)out->ptr;
    const uint64_t n_qk = in_dim / DS4GK_QK_K;
    std::vector<ds4gk_block_q8_K> xqk(n_qk);

    for (uint64_t t = 0; t < n_tok; t++) {
        const float *xt = xp + t * in_dim;
        float *ot = op + t * out_dim;
        ds4gk_quantize_row_q8_K(xt, xqk.data(), (int64_t)in_dim);
        for (uint64_t o = 0; o < out_dim; o++) {
            const uint8_t *row = base + o * row_bytes;
            ot[o] = (weight_type == 10)
                        ? ds4gk_vec_dot_q2_K_q8_K((int)in_dim,
                                                  (const ds4gk_block_q2_K *)row, xqk.data())
                        : ds4gk_vec_dot_iq2_xxs_q8_K((int)in_dim,
                                                     (const ds4gk_block_iq2_xxs *)row, xqk.data());
        }
    }
    return 1;
}

/* =========================================================================
 * Shared expert (the DS4 layer-FFN half that runs for every token).
 *
 * Host-side ports of ds4.c layer_shared_ffn_one / matvec_q8_0_pair_prequant /
 * swiglu / matvec_q8_0 / hc_post_one, reading the Q8_0 weights straight from
 * the model map and writing into the host-mapped tensors (same pattern as
 * ds4_gpu_routed_moe_one_tensor above).  The exact math:
 *
 *   gate  = dequant_q8_0(W_gate) @ x              (x re-quantized to Q8_0)
 *   up    = dequant_q8_0(W_up)   @ x
 *   if (clamp > 1e-6):  g = min(g, clamp);  u = clamp(u, -clamp, clamp)
 *   mid   = silu(g) * u                            (silu = x*sigmoid_stable(x))
 *   shared = dequant_q8_0(W_down) @ mid            (mid re-quantized to Q8_0)
 *   out_hc[dst][d] = post[dst] * (routed[d] + shared[d])
 *                    + sum_src comb[dst][src] * residual_hc[src][d]
 *
 * Weights are GGUF Q8_0 (f16 block scale + 32 int8 quants, 34 B/block); the
 * dots reuse the routed-MoE helper ds4gk_dot_q8_0_row so the reduction is
 * bit-identical with the other Q8_0 paths.  The `split` tensor follows the
 * HC convention [pre (n_hc) | post (n_hc) | comb (n_hc*n_hc)] with comb
 * addressed as [dst_hc, src_hc] (hc_post_one / kernel_dsv4_hc_expand).
 * ========================================================================= */
static int ds4_shared_gate_up_swiglu_q8_0_core(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              gate_offset,
        uint64_t              up_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        float                 clamp,
        int                   store_gate_up) {
    if (!mid || !x || !model_map ||
        (store_gate_up && (!gate || !up)) ||
        in_dim == 0 || out_dim == 0 ||
        !std::isfinite(clamp) || clamp < 0.0f) {
        return 0;
    }

    const uint64_t x_bytes = in_dim * sizeof(float);
    const uint64_t out_bytes = out_dim * sizeof(float);
    if (ds4_gpu_tensor_bytes(x) < x_bytes ||
        ds4_gpu_tensor_bytes(mid) < out_bytes ||
        (store_gate_up &&
         (ds4_gpu_tensor_bytes(gate) < out_bytes ||
          ds4_gpu_tensor_bytes(up) < out_bytes))) {
        return 0;
    }

    ds4_gpu_tensor *owned_gate = nullptr;
    ds4_gpu_tensor *owned_up = nullptr;
    if (!store_gate_up) {
        owned_gate = ds4_gpu_tensor_alloc(out_bytes);
        owned_up = ds4_gpu_tensor_alloc(out_bytes);
        gate = owned_gate;
        up = owned_up;
    }
    int ok = gate && up;
    if (ok) ok = ds4_gpu_matmul_q8_0_pair_tensor(
        gate, up, model_map, model_size, gate_offset, up_offset,
        in_dim, out_dim, out_dim, x, 1) != 0;
    if (ok) ok = ds4_gpu_swiglu_tensor(mid, gate, up, (uint32_t)out_dim,
                                        clamp, 1.0f) != 0;
    if (owned_up) ds4_gpu_tensor_free(owned_up);
    if (owned_gate) ds4_gpu_tensor_free(owned_gate);
    return ok;
}

int ds4_gpu_shared_gate_up_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *gate,
        ds4_gpu_tensor       *up,
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_shared_gate_up_swiglu_q8_0_core(gate, up, mid,
                                               model_map, model_size,
                                               gate_offset, up_offset,
                                               in_dim, out_dim, x, clamp, 1);
}

int ds4_gpu_shared_mid_swiglu_q8_0_tensor(
        ds4_gpu_tensor       *mid,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                gate_offset,
        uint64_t                up_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *x,
        float                   clamp) {
    return ds4_shared_gate_up_swiglu_q8_0_core(nullptr, nullptr, mid,
                                               model_map, model_size,
                                               gate_offset, up_offset,
                                               in_dim, out_dim, x, clamp, 0);
}

int ds4_gpu_shared_down_hc_expand_q8_0_tensor(
        ds4_gpu_tensor       *out_hc,
        ds4_gpu_tensor       *shared_out,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_offset,
        uint64_t                in_dim,
        uint64_t                out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t                n_embd,
        uint32_t                n_hc) {
    if (!out_hc || !shared_out || !model_map || !shared_mid || !routed_out ||
        !residual_hc || !split || n_embd == 0 || n_hc == 0 ||
        in_dim == 0 || out_dim != n_embd) {
        return 0;
    }

    const uint64_t embd_values = (uint64_t)n_embd;
    const uint64_t hc_values = (uint64_t)n_hc * embd_values;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    if (hc_values == 0 || mix_hc == 0 ||
        in_dim > UINT64_MAX / sizeof(float) ||
        embd_values > UINT64_MAX / sizeof(float) ||
        hc_values > UINT64_MAX / sizeof(float) ||
        mix_hc > UINT64_MAX / sizeof(float)) return 0;

    uint64_t rows = shared_mid->bytes / (in_dim * sizeof(float));
    rows = std::min(rows, shared_out->bytes / (embd_values * sizeof(float)));
    rows = std::min(rows, routed_out->bytes / (embd_values * sizeof(float)));
    rows = std::min(rows, residual_hc->bytes / (hc_values * sizeof(float)));
    rows = std::min(rows, split->bytes / (mix_hc * sizeof(float)));
    rows = std::min(rows, out_hc->bytes / (hc_values * sizeof(float)));
    if (rows == 0 || rows > UINT32_MAX) return 0;

    if (ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                   weight_offset, in_dim, out_dim,
                                   shared_mid, rows) == 0) return 0;
    return ds4_gpu_hc_expand_add_split_tensor(out_hc, routed_out, shared_out,
                                               residual_hc, split, n_embd, n_hc);
}

int ds4_gpu_shared_down_hc_expand_add_q8_0_tensor(
        ds4_gpu_tensor *out_hc,
        ds4_gpu_tensor *shared_out,
        const void *model_map,
        uint64_t model_size,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *shared_mid,
        const ds4_gpu_tensor *routed_out,
        const ds4_gpu_tensor *routed_add,
        const ds4_gpu_tensor *residual_hc,
        const ds4_gpu_tensor *split,
        uint32_t n_embd,
        uint32_t n_hc) {
    if (!out_hc || !shared_out || !model_map || !shared_mid || !routed_out ||
        !routed_add || !residual_hc || !split || n_embd == 0 || n_hc == 0 ||
        in_dim == 0 || out_dim != n_embd ||
        out_dim > UINT64_MAX / sizeof(float)) return 0;

    const uint64_t embd_values = (uint64_t)n_embd;
    const uint64_t hc_values = (uint64_t)n_hc * embd_values;
    const uint64_t mix_hc = 2ull * n_hc + (uint64_t)n_hc * n_hc;
    if (hc_values == 0 || mix_hc == 0 || in_dim > UINT64_MAX / sizeof(float) ||
        hc_values > UINT64_MAX / sizeof(float) ||
        mix_hc > UINT64_MAX / sizeof(float)) return 0;

    uint64_t rows = shared_mid->bytes / (in_dim * sizeof(float));
    rows = std::min(rows, shared_out->bytes / (embd_values * sizeof(float)));
    rows = std::min(rows, routed_out->bytes / (embd_values * sizeof(float)));
    rows = std::min(rows, routed_add->bytes / (embd_values * sizeof(float)));
    rows = std::min(rows, residual_hc->bytes / (hc_values * sizeof(float)));
    rows = std::min(rows, split->bytes / (mix_hc * sizeof(float)));
    rows = std::min(rows, out_hc->bytes / (hc_values * sizeof(float)));
    if (rows == 0 || rows > UINT32_MAX || rows > UINT64_MAX / embd_values ||
        rows * embd_values > UINT32_MAX) return 0;

    ds4_gpu_tensor *routed_sum = ds4_gpu_tensor_alloc(rows * embd_values * sizeof(float));
    if (!routed_sum) return 0;
    int ok = ds4_gpu_matmul_q8_0_tensor(shared_out, model_map, model_size,
                                        weight_offset, in_dim, out_dim,
                                        shared_mid, rows) != 0;
    if (ok) ok = ds4_gpu_add_tensor(routed_sum, routed_out, routed_add,
                                    (uint32_t)(rows * embd_values)) != 0;
    if (ok) ok = ds4_gpu_hc_expand_add_split_tensor(
        out_hc, shared_out, routed_sum, residual_hc, split, n_embd, n_hc) != 0;
    ds4_gpu_tensor_free(routed_sum);
    return ok;
}

/* =========================================================================
 * DS4 KV compressor (host-side; mirrors ds4_cuda.cu compressor kernels).
 *
 * Layout used by every function here (matches the CUDA/Metal production path):
 *   width      = (ratio == 4 ? 2 : 1) * head_dim
 *   state_rows = (ratio == 4 ? 2 : 1) * ratio
 *   state_kv / state_score : [state_rows][width] f32
 *   comp_cache             : [n_comp][head_dim] f32 (the exact layout consumed
 *                            by ds4_gpu_attention_decode_heads_tensor)
 * Ratio-4 layers keep two lanes: rows [0,4) are the attention lane, rows
 * [4,8) the indexer lane.  The prefill pool for comp row c mixes the previous
 * window's attention lane (kv[(c-1)*4..c*4), dim d) with the current window's
 * indexer lane (kv[c*4..c*4+4), dim head_dim + d).  Pooling is a per-dimension
 * softmax over the candidate scores followed by a weighted average of the KV
 * values (max-subtraction for numeric stability, like the CUDA kernels).
 *
 * The APE weight is added to the score rows (ape_type 0 = f32, 1 = f16).
 * ========================================================================= */

/* Store a batch of projected kv/score rows into the rolling compressor state.
 * Rows are mapped to state rows by (pos0 + t) % ratio; ratio-4 layers store
 * the window in the second lane (rows [ratio, 2*ratio)). */
int ds4_gpu_compressor_store_batch_tensor(
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens) {
    if (!kv || !sc || !state_kv || !state_score || !model_map ||
        head_dim == 0 || ratio == 0 || ratio > 128u || n_tokens == 0 ||
        (ape_type != 0u && ape_type != 1u) || n_tokens > 65535u) return 0;
    const uint32_t width = (ratio == 4u ? 2u : 1u) * head_dim;
    const uint32_t state_rows = (ratio == 4u ? 2u : 1u) * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes)
        return 0;
    VkBuffer skbuf, ssbuf, kvbuf, scbuf, abuf;
    VkDeviceSize skoff, ssoff, kvoff, scoff, aoff, arange;
    if (!find_tensor_buffer(state_kv, skbuf, skoff) ||
        !find_tensor_buffer(state_score, ssbuf, ssoff) ||
        !find_tensor_buffer(kv, kvbuf, kvoff) || !find_tensor_buffer(sc, scbuf, scoff) ||
        !find_model_buffer(ape_offset, ape_bytes, abuf, aoff, arange)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((skoff | ssoff | kvoff | scoff | aoff) % align) != 0) return 0;
    const uint64_t work = (uint64_t)n_tokens * width;
    const uint32_t groups = (uint32_t)((work + 255u) / 256u);
    if (groups == 0 || groups > 65535u) return 0;
    auto &ctx = get_cmd_ctx(); const bool resume_recording = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[5] = {
        {skbuf, skoff, (VkDeviceSize)state_kv->bytes},
        {ssbuf, ssoff, (VkDeviceSize)state_score->bytes},
        {kvbuf, kvoff, (VkDeviceSize)kv->bytes},
        {scbuf, scoff, (VkDeviceSize)sc->bytes},
        {abuf, aoff, (VkDeviceSize)ape_bytes}};
    struct { uint32_t width, ratio, pos0, n_tokens, ape_type, r0, r1, r2; }
        pc = {width, ratio, pos0, n_tokens, ape_type, 0, 0, 0};
    DS4_VK_TRACE_KERNEL("compressor_store");
    return record_simple_shader("compressor_store", &pc, sizeof(pc), bufs, 5,
                                groups, 1, 1, resume_recording);
}

static int compressor_clear_vk(ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
                               uint32_t count, float kv_value, float score_value) {
    if (!state_kv || !state_score || count == 0 || count > 0xffffffffu - 255u) return 0;
    VkBuffer state_kv_buf, state_score_buf;
    VkDeviceSize state_kv_off, state_score_off;
    if (!find_tensor_buffer(state_kv, state_kv_buf, state_kv_off) ||
        !find_tensor_buffer(state_score, state_score_buf, state_score_off)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((state_kv_off | state_score_off) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[2] = {
        {state_kv_buf, state_kv_off, (VkDeviceSize)state_kv->bytes},
        {state_score_buf, state_score_off, (VkDeviceSize)state_score->bytes}};
    struct { uint32_t count; float kv_value; float score_value; }
        pc = {count, kv_value, score_value};
    return record_simple_shader("compressor_clear", &pc, sizeof(pc), bufs, 2,
                                (count + 255u) / 256u, 1, 1, resume);
}

static int compressor_set_rows_vk(
        const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc,
        ds4_gpu_tensor *state_kv, ds4_gpu_tensor *state_score,
        VkBuffer ape_buf, VkDeviceSize ape_off, uint32_t ape_type,
        uint32_t width, uint32_t ratio, uint32_t pos0,
        uint32_t src0, uint32_t dst0, uint32_t rows) {
    if (rows == 0 || width == 0 || width > 4096u || ratio == 0 || ratio > 128u ||
        rows > 65535u || !shader_f32_domain(rows, width, 1)) return 0;
    VkBuffer kv_buf, sc_buf, state_kv_buf, state_score_buf;
    VkDeviceSize kv_off, sc_off, state_kv_off, state_score_off;
    if (!find_tensor_buffer(kv, kv_buf, kv_off) ||
        !find_tensor_buffer(sc, sc_buf, sc_off) ||
        !find_tensor_buffer(state_kv, state_kv_buf, state_kv_off) ||
        !find_tensor_buffer(state_score, state_score_buf, state_score_off)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((kv_off | sc_off | state_kv_off | state_score_off | ape_off) % align) != 0)
        return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[5] = {
        {state_kv_buf, state_kv_off, (VkDeviceSize)state_kv->bytes},
        {state_score_buf, state_score_off, (VkDeviceSize)state_score->bytes},
        {kv_buf, kv_off, (VkDeviceSize)kv->bytes},
        {sc_buf, sc_off, (VkDeviceSize)sc->bytes},
        {ape_buf, ape_off, VK_WHOLE_SIZE}};
    struct { uint32_t width, ratio, pos0, src0, dst0, rows, ape_type, reserved; }
        pc = {width, ratio, pos0, src0, dst0, rows, ape_type, 0};
    return record_simple_shader("compressor_set_rows", &pc, sizeof(pc), bufs, 5,
                                (uint32_t)(((uint64_t)rows * width + 255u) / 256u),
                                1, 1, resume);
}

static int compressor_pool_vk(
        ds4_gpu_tensor *out, const ds4_gpu_tensor *kv, const ds4_gpu_tensor *sc,
        const ds4_gpu_tensor *state_kv, const ds4_gpu_tensor *state_score,
        VkBuffer ape_buf, VkDeviceSize ape_off, uint32_t ape_type,
        uint32_t head_dim, uint32_t ratio, uint32_t pos0, uint32_t n_comp,
        bool replay) {
    VkBuffer out_buf, kv_buf, sc_buf, state_kv_buf, state_score_buf;
    VkDeviceSize out_off, kv_off, sc_off, state_kv_off, state_score_off;
    if (!find_tensor_buffer(out, out_buf, out_off) ||
        !find_tensor_buffer(kv, kv_buf, kv_off) ||
        !find_tensor_buffer(sc, sc_buf, sc_off) ||
        !find_tensor_buffer(state_kv, state_kv_buf, state_kv_off) ||
        !find_tensor_buffer(state_score, state_score_buf, state_score_off)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((out_off | kv_off | sc_off | state_kv_off | state_score_off | ape_off) % align) != 0)
        return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[6] = {
        {out_buf, out_off, (VkDeviceSize)out->bytes},
        {kv_buf, kv_off, (VkDeviceSize)kv->bytes},
        {sc_buf, sc_off, (VkDeviceSize)sc->bytes},
        {state_kv_buf, state_kv_off, (VkDeviceSize)state_kv->bytes},
        {state_score_buf, state_score_off, (VkDeviceSize)state_score->bytes},
        {ape_buf, ape_off, VK_WHOLE_SIZE}};
    struct { uint32_t head_dim, ratio, pos0, n_comp, ape_type, replay, r0, r1; }
        pc = {head_dim, ratio, pos0, n_comp, ape_type, replay ? 1u : 0u, 0, 0};
    return record_simple_shader("compressor_pool", &pc, sizeof(pc), bufs, 6,
                                (head_dim + 255u) / 256u, n_comp, 1, resume);
}

static int compressor_pool_state_vk(ds4_gpu_tensor *out,
                                    const ds4_gpu_tensor *state_kv,
                                    const ds4_gpu_tensor *state_score,
                                    uint32_t head_dim, uint32_t ratio) {
    VkBuffer out_buf, state_kv_buf, state_score_buf;
    VkDeviceSize out_off, state_kv_off, state_score_off;
    if (!find_tensor_buffer(out, out_buf, out_off) ||
        !find_tensor_buffer(state_kv, state_kv_buf, state_kv_off) ||
        !find_tensor_buffer(state_score, state_score_buf, state_score_off)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((out_off | state_kv_off | state_score_off) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[3] = {
        {out_buf, out_off, (VkDeviceSize)out->bytes},
        {state_kv_buf, state_kv_off, (VkDeviceSize)state_kv->bytes},
        {state_score_buf, state_score_off, (VkDeviceSize)state_score->bytes}};
    struct { uint32_t head_dim, ratio; } pc = {head_dim, ratio};
    return record_simple_shader("compressor_pool_state", &pc, sizeof(pc), bufs, 3,
                                (head_dim + 255u) / 256u, 1, 1, resume);
}

static int compressor_shift_ratio4_vk(ds4_gpu_tensor *state_kv,
                                      ds4_gpu_tensor *state_score,
                                      uint32_t width) {
    VkBuffer state_kv_buf, state_score_buf;
    VkDeviceSize state_kv_off, state_score_off;
    if (!find_tensor_buffer(state_kv, state_kv_buf, state_kv_off) ||
        !find_tensor_buffer(state_score, state_score_buf, state_score_off)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && ((state_kv_off | state_score_off) % align) != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[2] = {
        {state_kv_buf, state_kv_off, (VkDeviceSize)state_kv->bytes},
        {state_score_buf, state_score_off, (VkDeviceSize)state_score->bytes}};
    return record_simple_shader("compressor_shift_ratio4", &width, sizeof(width), bufs, 2,
                                (uint32_t)(((uint64_t)width * 4u + 255u) / 256u),
                                1, 1, resume);
}

static int compressor_rope_stride_vk(ds4_gpu_tensor *x, uint32_t n_tok,
                                     uint32_t head_dim, uint32_t n_rot,
                                     uint32_t pos0, uint32_t pos_stride,
                                     uint32_t n_ctx_orig, float freq_base,
                                     float freq_scale, float ext_factor,
                                     float attn_factor, float beta_fast,
                                     float beta_slow) {
    if (!x || n_tok == 0 || head_dim == 0 || n_rot == 0 || n_rot > head_dim ||
        (n_rot & 1u) != 0 || n_tok > 65535u || freq_base <= 0.0f ||
        freq_scale <= 0.0f) return 0;
    if ((uint64_t)n_tok > UINT64_MAX / head_dim ||
        (uint64_t)n_tok * head_dim > UINT64_MAX / sizeof(float) ||
        (uint64_t)n_tok * head_dim * sizeof(float) > x->bytes) return 0;
    const uint64_t pairs = (uint64_t)n_tok * (n_rot / 2u);
    if (pairs == 0 || pairs > UINT32_MAX - 255u ||
        (pairs + 255u) / 256u > 65535u) return 0;
    VkBuffer x_buf; VkDeviceSize x_off;
    if (!find_tensor_buffer(x, x_buf, x_off)) return 0;
    const VkDeviceSize align = g_vk.caps.min_storage_buffer_offset_alignment;
    if (align && x_off % align != 0) return 0;
    auto &ctx = get_cmd_ctx();
    const bool resume = ctx.recording;
    if (ctx.recording && ctx.command_count != 0 && !submit_and_wait()) return 0;
    if (!ctx.recording && !begin_cmd()) return 0;
    VkDescriptorBufferInfo bufs[1] = {{x_buf, x_off, (VkDeviceSize)x->bytes}};
    struct Push {
        uint32_t n_tok, head_dim, n_rot, pos0, pos_stride, n_ctx_orig;
        int32_t inverse;
        float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow;
    } pc = {n_tok, head_dim, n_rot, pos0, pos_stride, n_ctx_orig, 0,
            freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow};
    return record_simple_shader("compressor_rope_stride", &pc, sizeof(pc), bufs, 1,
                                (uint32_t)((pairs + 255u) / 256u), 1, 1, resume);
}

/* Streaming decode update for one token.  Stores the projected kv/score row
 * into the rolling state and, at ratio boundaries ((pos + 1) % ratio == 0),
 * pools the state into comp_cache[comp_row], applies RMS norm + RoPE tail and
 * (ratio-4) shifts the state for the next window.  Matches ds4_cuda.cu
 * ds4_gpu_compressor_update_tensor exactly. */
int ds4_gpu_compressor_update_tensor(
        const ds4_gpu_tensor *kv_cur,
        const ds4_gpu_tensor *sc_cur,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        ds4_gpu_tensor       *comp_cache,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos,
        uint32_t                comp_row,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps,
        bool                    state_already_stored,
        bool                    decode_one_token,
        bool                    defer_finalize) {
    DS4_VK_TRACE_KERNEL("compressor_update");
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache || !model_map ||
        head_dim == 0 || ratio == 0 || ratio > 128u || n_rot > head_dim ||
        (n_rot & 1u) != 0 || (ape_type != 0u && ape_type != 1u) || norm_type != 0u)
        return 0;
    const uint32_t width = (ratio == 4u ? 2u : 1u) * head_dim;
    const uint32_t state_rows = (ratio == 4u ? 2u : 1u) * ratio;
    const bool emit = ((pos + 1u) % ratio) == 0u;
    const uint64_t ape_bytes = (uint64_t)width * ratio * (ape_type == 1u ? 2u : 4u);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t row_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || row_bytes > model_size - norm_offset ||
        kv_cur->bytes < (uint64_t)width * sizeof(float) ||
        sc_cur->bytes < (uint64_t)width * sizeof(float) ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && ((uint64_t)comp_row + 1u) * row_bytes > comp_cache->bytes)) return 0;
    set_model_map_identity(model_map, model_size);
    VkBuffer ape_buf; VkDeviceSize ape_off, ape_range;
    if (!find_model_buffer(ape_offset, ape_bytes, ape_buf, ape_off, ape_range)) return 0;
    if (!state_already_stored && !ds4_gpu_compressor_store_batch_tensor(
            kv_cur, sc_cur, state_kv, state_score, model_map, model_size,
            ape_offset, ape_type, head_dim, ratio, pos, 1)) return 0;
    if (!emit) return 1;
    ds4_gpu_tensor *row = ds4_gpu_tensor_view(
        comp_cache, (uint64_t)comp_row * row_bytes, row_bytes);
    if (!row) return 0;
    int ok = compressor_pool_state_vk(row, state_kv, state_score, head_dim, ratio);
    /* A successful update always leaves the emitted row and rolling state
     * finalized, including when called by the fused projection path. */
    if (ok)
        ok = ds4_gpu_rms_norm_weight_rows_tensor(row, row, model_map, model_size,
                                                  norm_offset, head_dim, 1, rms_eps);
    if (ok && n_rot != 0)
        ok = ds4_gpu_rope_tail_tensor(row, 1, 1, head_dim, n_rot,
                                      pos + 1u - ratio, n_ctx_orig, false,
                                      freq_base, freq_scale, ext_factor, attn_factor,
                                      beta_fast, beta_slow);
    if (ok && ratio == 4u)
        ok = compressor_shift_ratio4_vk(state_kv, state_score, width);
    ds4_gpu_tensor_free(row);
    (void)decode_one_token;
    (void)defer_finalize;
    return ok;
#if 0
    (void)decode_one_token;
    (void)defer_finalize;
    if (!kv_cur || !sc_cur || !state_kv || !state_score || !comp_cache ||
        !model_map || head_dim == 0 || ratio == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t emit = ((pos + 1u) % ratio) == 0u ? 1u : 0u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes =
        (uint64_t)(comp_row + (emit ? 1u : 0u)) * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv_cur->bytes < kv_bytes || sc_cur->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (emit && comp_cache->bytes < comp_bytes)) {
        return 0;
    }

    if (!state_already_stored) {
        if (!ds4_gpu_compressor_store_batch_tensor(
                kv_cur, sc_cur, state_kv, state_score,
                model_map, model_size, ape_offset, ape_type,
                head_dim, ratio, pos, 1)) {
            return 0;
        }
    }
    if (!emit) return 1;

    ds4_gpu_tensor comp_row_view;
    comp_row_view.ptr = (char *)comp_cache->ptr +
        (uint64_t)comp_row * head_dim * sizeof(float);
    comp_row_view.bytes = (uint64_t)head_dim * sizeof(float);
    comp_row_view.owner = 0;
    comp_row_view.device_id = comp_cache->device_id;

    compressor_pool_state((float *)comp_row_view.ptr,
                          (const float *)state_kv->ptr,
                          (const float *)state_score->ptr,
                          head_dim, ratio);
    if (!ds4_gpu_rms_norm_weight_rows_tensor(
            &comp_row_view, &comp_row_view, model_map, model_size,
            norm_offset, head_dim, 1, rms_eps)) {
        return 0;
    }
    if (!ds4_gpu_rope_tail_tensor(
            &comp_row_view, 1, 1, head_dim, n_rot,
            pos + 1u - ratio, n_ctx_orig, false,
            freq_base, freq_scale, ext_factor, attn_factor,
            beta_fast, beta_slow)) {
        return 0;
    }
    if (ratio == 4u) {
        compressor_shift_ratio4((float *)state_kv->ptr,
                                (float *)state_score->ptr, width);
    }
    return 1;
#endif
}

/* Prefill compression: pool the kv/sc batch into comp_cache (n_comp = n_tokens
 * / ratio rows), initialize the rolling state from the rows after the last
 * complete window, apply RMS norm, stride-ratio RoPE tail and optional fp8
 * quantization.  Matches ds4_cuda.cu ds4_gpu_compressor_prefill_tensor. */
int ds4_gpu_compressor_prefill_tensor(
        ds4_gpu_tensor       *comp_cache,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                norm_offset,
        uint32_t                norm_type,
        uint32_t                head_dim,
        uint32_t                ratio,
        uint32_t                pos0,
        uint32_t                n_tokens,
        uint32_t                n_rot,
        uint32_t                n_ctx_orig,
        bool                    quantize_fp8,
        float                   freq_base,
        float                   freq_scale,
        float                   ext_factor,
        float                   attn_factor,
        float                   beta_fast,
        float                   beta_slow,
        float                   rms_eps) {
    DS4_VK_TRACE_KERNEL("compressor_prefill");
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || ratio > 128u || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u || n_tokens > 65535u)
    {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr, "ds4: [dbg] compressor_prefill invalid args head=%u ratio=%u tokens=%u rot=%u ape=%u norm=%u\n",
                    head_dim, ratio, n_tokens, n_rot, ape_type, norm_type);
        return 0;
    }
    const uint32_t width = (ratio == 4u ? 2u : 1u) * head_dim;
    const uint32_t state_rows = (ratio == 4u ? 2u : 1u) * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t ape_bytes = (uint64_t)width * ratio * (ape_type == 1u ? 2u : 4u);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp != 0 && comp_cache->bytes < comp_bytes)) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr, "ds4: [dbg] compressor_prefill bounds comp=%llu/%llu state=%llu/%llu kv=%llu/%llu\n",
                    (unsigned long long)comp_cache->bytes, (unsigned long long)comp_bytes,
                    (unsigned long long)state_kv->bytes, (unsigned long long)state_bytes,
                    (unsigned long long)kv->bytes, (unsigned long long)kv_bytes);
        return 0;
    }
    set_model_map_identity(model_map, model_size);
    VkBuffer ape_buf; VkDeviceSize ape_off, ape_range;
    if (!find_model_buffer(ape_offset, ape_bytes, ape_buf, ape_off, ape_range)) {
        if (getenv("DS4_VULKAN_DEBUG")) fprintf(stderr, "ds4: [dbg] compressor_prefill ape upload failed\n");
        return 0;
    }
    if (!compressor_clear_vk(state_kv, state_score, state_rows * width,
                             0.0f, -INFINITY)) {
        if (getenv("DS4_VULKAN_DEBUG")) fprintf(stderr, "ds4: [dbg] compressor_prefill clear failed\n");
        return 0;
    }
    int ok = 1;
    if (ratio == 4u) {
        if (cutoff >= ratio)
            ok = compressor_set_rows_vk(kv, sc, state_kv, state_score, ape_buf, ape_off,
                                        ape_type, width, ratio, pos0,
                                        cutoff - ratio, 0, ratio);
        if (ok && rem != 0)
            ok = compressor_set_rows_vk(kv, sc, state_kv, state_score, ape_buf, ape_off,
                                        ape_type, width, ratio, pos0,
                                        cutoff, ratio, rem);
    } else if (rem != 0) {
        ok = compressor_set_rows_vk(kv, sc, state_kv, state_score, ape_buf, ape_off,
                                    ape_type, width, ratio, pos0,
                                    cutoff, 0, rem);
    }
    if (ok && n_comp != 0)
        ok = compressor_pool_vk(comp_cache, kv, sc, state_kv, state_score,
                                ape_buf, ape_off, ape_type, head_dim, ratio, pos0,
                                n_comp, false);
    if (!ok && getenv("DS4_VULKAN_DEBUG")) fprintf(stderr, "ds4: [dbg] compressor_prefill pool/state failed\n");
    if (!ok) return 0;
    if (ok && n_comp != 0)
        ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache, model_map,
                                                  model_size, norm_offset, head_dim,
                                                  n_comp, rms_eps);
    if (!ok && getenv("DS4_VULKAN_DEBUG")) fprintf(stderr, "ds4: [dbg] compressor_prefill norm failed\n");
    if (!ok) return 0;
    if (ok && n_comp != 0 && n_rot != 0)
        ok = compressor_rope_stride_vk(comp_cache, n_comp, head_dim, n_rot, pos0, ratio,
                                       n_ctx_orig, freq_base, freq_scale, ext_factor,
                                       attn_factor, beta_fast, beta_slow);
    if (!ok && getenv("DS4_VULKAN_DEBUG")) fprintf(stderr, "ds4: [dbg] compressor_prefill rope failed\n");
    if (!ok) return 0;
    if (ok && n_comp != 0 && quantize_fp8)
        ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot);
    if (!ok && getenv("DS4_VULKAN_DEBUG")) fprintf(stderr, "ds4: [dbg] compressor_prefill fp8 failed\n");
    return ok;
#if 0
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || ratio == 0 || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u) {
        return 0;
    }

    const uint32_t coff = ratio == 4u ? 2u : 1u;
    const uint32_t width = coff * head_dim;
    const uint32_t state_rows = coff * ratio;
    const uint32_t n_comp = n_tokens / ratio;
    const uint32_t cutoff = n_comp * ratio;
    const uint32_t rem = n_tokens - cutoff;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);

    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        (n_comp && comp_cache->bytes < comp_bytes)) {
        return 0;
    }

    const uint64_t state_n = (uint64_t)state_rows * width;
    memset(state_kv->ptr, 0, (size_t)(state_n * sizeof(float)));
    float *ss = (float *)state_score->ptr;
    for (uint64_t i = 0; i < state_n; i++) ss[i] = -INFINITY;

    if (ratio == 4u) {
        if (cutoff >= ratio) {
            compressor_set_rows((float *)state_kv->ptr, ss,
                                (const float *)kv->ptr, (const float *)sc->ptr,
                                model_map, ape_offset, ape_type,
                                width, ratio, pos0,
                                cutoff - ratio, 0, ratio);
        }
        if (rem != 0) {
            compressor_set_rows((float *)state_kv->ptr, ss,
                                (const float *)kv->ptr, (const float *)sc->ptr,
                                model_map, ape_offset, ape_type,
                                width, ratio, pos0,
                                cutoff, ratio, rem);
        }
    } else if (rem != 0) {
        compressor_set_rows((float *)state_kv->ptr, ss,
                            (const float *)kv->ptr, (const float *)sc->ptr,
                            model_map, ape_offset, ape_type,
                            width, ratio, pos0,
                            cutoff, 0, rem);
    }
    if (n_comp != 0) {
        compressor_prefill_pool((float *)comp_cache->ptr,
                                (const float *)kv->ptr, (const float *)sc->ptr,
                                (const float *)state_kv->ptr, ss,
                                model_map, ape_offset, ape_type,
                                head_dim, ratio, pos0, n_comp, 0);
        if (!ds4_gpu_rms_norm_weight_rows_tensor(
                comp_cache, comp_cache, model_map, model_size,
                norm_offset, head_dim, n_comp, rms_eps)) {
            return 0;
        }
        if (n_rot != 0) {
            /* CUDA's rope_tail_kernel uses pos_stride = ratio, i.e. comp row c
             * sits at absolute position pos0 + c * ratio. */
            for (uint32_t c = 0; c < n_comp; c++) {
                ds4_gpu_tensor row_view;
                row_view.ptr = (char *)comp_cache->ptr +
                    (uint64_t)c * head_dim * sizeof(float);
                row_view.bytes = (uint64_t)head_dim * sizeof(float);
                row_view.owner = 0;
                row_view.device_id = comp_cache->device_id;
                if (!ds4_gpu_rope_tail_tensor(
                        &row_view, 1, 1, head_dim, n_rot,
                        pos0 + c * ratio, n_ctx_orig, false,
                        freq_base, freq_scale, ext_factor, attn_factor,
                        beta_fast, beta_slow)) {
                    return 0;
                }
            }
        }
        if (quantize_fp8 &&
            !ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot)) {
            return 0;
        }
    }
    return 1;
#endif
}

int ds4_gpu_compressor_prefill_ratio4_replay_tensor(
        ds4_gpu_tensor *comp_cache, ds4_gpu_tensor *state_kv,
        ds4_gpu_tensor *state_score, const ds4_gpu_tensor *kv,
        const ds4_gpu_tensor *sc, const void *model_map, uint64_t model_size,
        uint64_t ape_offset, uint32_t ape_type, uint64_t norm_offset,
        uint32_t norm_type, uint32_t head_dim, uint32_t pos0,
        uint32_t n_tokens, uint32_t n_rot, uint32_t n_ctx_orig,
        bool quantize_fp8, float freq_base, float freq_scale, float ext_factor,
        float attn_factor, float beta_fast, float beta_slow, float rms_eps) {
    DS4_VK_TRACE_KERNEL("compressor_prefill_ratio4_replay");
    if (!comp_cache || !state_kv || !state_score || !kv || !sc || !model_map ||
        head_dim == 0 || n_tokens == 0 || (n_tokens & 3u) != 0 ||
        (pos0 & 3u) != 0 || n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u || n_tokens > 65535u)
        return 0;
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint32_t n_comp = n_tokens / ratio;
    const uint64_t ape_bytes = (uint64_t)width * ratio * (ape_type == 1u ? 2u : 4u);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_tokens * width * sizeof(float);
    const uint64_t comp_bytes = (uint64_t)n_comp * head_dim * sizeof(float);
    const uint64_t norm_bytes = (uint64_t)head_dim * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        norm_offset > model_size || norm_bytes > model_size - norm_offset ||
        kv->bytes < kv_bytes || sc->bytes < kv_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes ||
        comp_cache->bytes < comp_bytes) return 0;
    set_model_map_identity(model_map, model_size);
    VkBuffer ape_buf; VkDeviceSize ape_off, ape_range;
    if (!find_model_buffer(ape_offset, ape_bytes, ape_buf, ape_off, ape_range)) return 0;
    int ok = compressor_pool_vk(comp_cache, kv, sc, state_kv, state_score,
                                ape_buf, ape_off, ape_type, head_dim, ratio, pos0,
                                n_comp, true);
    if (ok) ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache, model_map,
                                                      model_size, norm_offset, head_dim,
                                                      n_comp, rms_eps);
    if (ok && n_rot != 0)
        ok = compressor_rope_stride_vk(comp_cache, n_comp, head_dim, n_rot, pos0, ratio,
                                       n_ctx_orig, freq_base, freq_scale, ext_factor,
                                       attn_factor, beta_fast, beta_slow);
    if (ok && quantize_fp8)
        ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot);
    if (ok && !compressor_clear_vk(state_kv, state_score, state_rows * width,
                                   0.0f, -INFINITY)) ok = 0;
    if (ok)
        ok = compressor_set_rows_vk(kv, sc, state_kv, state_score, ape_buf, ape_off,
                                    ape_type, width, ratio, pos0,
                                    n_tokens - ratio, 0, ratio);
    return ok;
}

/* ---- matmul_f16_pair_compressor_store_tensor (host-side fused path) ----
 *
 * Fuses the paired F16 compressor projections with the rolling compressor
 * state store for one decode token:
 *
 *   out_kv[o]    = sum_i W_kv[o][i] * x[i]        (o < width, i < in_dim)
 *   out_score[o] = sum_i W_score[o][i] * x[i]
 *   state_kv / state_score row = out_kv / out_score + APE(phase = pos % ratio)
 *
 * The weights are IEEE-half matrices (2 bytes/element, row-major
 * [width][in_dim]) at weight_kv_offset / weight_score_offset; the APE
 * tensor is laid out [ratio][width] (ape_type 0 = f32, 1 = f16), exactly
 * like the CUDA compressor_store_kernel.  Row mapping matches
 * ds4_gpu_compressor_store_batch_tensor: ratio-4 layers keep the current
 * window in the second lane (rows [ratio, 2*ratio)).
 *
 * The projections and state update are composed from Vulkan dispatches. */
int ds4_gpu_matmul_f16_pair_compressor_store_tensor(
        ds4_gpu_tensor       *out_kv,
        ds4_gpu_tensor       *out_score,
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                weight_kv_offset,
        uint64_t                weight_score_offset,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint64_t                in_dim,
        uint32_t                width,
        const ds4_gpu_tensor *x,
        uint32_t                ratio,
        uint32_t                pos) {
    if (!out_kv || !out_score || !state_kv || !state_score || !model_map || !x ||
        in_dim == 0 || width == 0 || ratio == 0 || ratio > 128u ||
        in_dim > UINT32_MAX || width > UINT32_MAX ||
        (ape_type != 0u && ape_type != 1u)) {
        return -1;
    }
    if (in_dim > UINT64_MAX / width ||
        in_dim * width > UINT64_MAX / 2u) return -1;
    const uint32_t coff = ratio == 4u ? 2u : 1u;
    if (coff > UINT32_MAX / ratio) return -1;
    const uint32_t state_rows = coff * ratio;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t weight_bytes = in_dim * (uint64_t)width * 2u;
    const uint64_t out_bytes = (uint64_t)width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)width * ratio * elem_ape;
    if (weight_kv_offset > model_size ||
        weight_bytes > model_size - weight_kv_offset ||
        weight_score_offset > model_size ||
        weight_bytes > model_size - weight_score_offset ||
        ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        in_dim > UINT64_MAX / sizeof(float) ||
        in_dim * sizeof(float) > x->bytes ||
        out_kv->bytes < out_bytes || out_score->bytes < out_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return -1;
    }

    set_model_map_identity(model_map, model_size);
    if (!ds4_gpu_matmul_f16_tensor(out_kv, model_map, model_size,
                                   weight_kv_offset, in_dim, width, x, 1) ||
        !ds4_gpu_matmul_f16_tensor(out_score, model_map, model_size,
                                   weight_score_offset, in_dim, width, x, 1))
        return 0;
    return ds4_gpu_compressor_store_batch_tensor(
        out_kv, out_score, state_kv, state_score, model_map, model_size,
        ape_offset, ape_type, width / coff, ratio, pos, 1) ? 1 : 0;
}

/* ---- matmul_q8_0_f16_out_tensor (host-side, f16 output) ----
 *
 * out_h[o + t*out_dim] = sum_i deq_q8(W[o][i]) * x[i + t*in_dim]
 * for o in [0, out_dim), t in [0, n_tok), with the same Q8_0 dequant math
 * as the verified matmul_q8_0 kernel (GGUF 34 B/block: f16 scale + 32
 * int8, double accumulation, partial last block) but the result is stored
 * as an IEEE half (2 B/element) instead of f32 — the engine's
 * batch_q_half buffer for the shared-expert down projection.
 *
 * This is host-side (tensor->ptr is host-mapped), so no command buffer is
 * involved.  Returns 1 on success, -1 on invalid arguments / out-of-range
 * offsets or tensor sizes.
 */
int ds4_gpu_matmul_q8_0_f16_out_tensor(
    ds4_gpu_tensor *out_h, const void *model_map, uint64_t model_size,
    uint64_t weight_offset, uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x, uint64_t n_tok)
{
    (void)out_h; (void)model_map; (void)model_size; (void)weight_offset;
    (void)in_dim; (void)out_dim; (void)x; (void)n_tok;
    return 0;
}

/* ---- matmul_f16_pair_tensor (host-side paired f16 projections) ----
 *
 *   out_a[o + t*out_dim] = sum_i deq_f16(Wa[o][i]) * x[i + t*in_dim]
 *   out_b[o + t*out_dim] = sum_i deq_f16(Wb[o][i]) * x[i + t*in_dim]
 *
 * Both weight matrices are IEEE half (2 B/element), row-major
 * [out_dim][in_dim], like the verified matmul_f16 path; outputs are f32.
 * Used for the compressor / indexer KV + score paired projections (n_tok
 * may be > 1, e.g. the ratio-4 prefill tail).  Host-side, no command
 * buffer.  Returns 1 on success, -1 on invalid arguments / out-of-range
 * offsets or tensor sizes.
 */
int ds4_gpu_matmul_f16_pair_tensor(
    ds4_gpu_tensor *out_a, ds4_gpu_tensor *out_b,
    const void *model_map, uint64_t model_size,
    uint64_t weight_a_offset, uint64_t weight_b_offset,
    uint64_t in_dim, uint64_t out_dim,
    const ds4_gpu_tensor *x, uint64_t n_tok)
{
    DS4_VK_TRACE_KERNEL("matmul_f16_pair");
    if (!out_a || !out_b || !model_map || !x ||
        in_dim == 0 || out_dim == 0 || n_tok == 0)
        return -1;
    if (!out_a->ptr || !out_b->ptr || !x->ptr) return -1;

    if (in_dim > UINT64_MAX / out_dim || in_dim * out_dim > UINT64_MAX / 2u)
        return -1;
    const uint64_t weight_bytes = in_dim * out_dim * 2u;  /* f16: 2 B/elem */
    /* Never read past the model mmap (SIGBUS guard), never write past
     * tensor bytes. */
    if (weight_a_offset > model_size ||
        weight_bytes > model_size - weight_a_offset)
        return -1;
    if (weight_b_offset > model_size ||
        weight_bytes > model_size - weight_b_offset)
        return -1;
    if (n_tok > UINT64_MAX / in_dim ||
        in_dim * n_tok > UINT64_MAX / sizeof(float) ||
        in_dim * n_tok * sizeof(float) > x->bytes)
        return -1;
    if (n_tok > UINT64_MAX / out_dim ||
        out_dim * n_tok > UINT64_MAX / sizeof(float) ||
        out_dim * n_tok * sizeof(float) > out_a->bytes)
        return -1;
    if (n_tok > UINT64_MAX / out_dim ||
        out_dim * n_tok > UINT64_MAX / sizeof(float) ||
        out_dim * n_tok * sizeof(float) > out_b->bytes)
        return -1;

    if (!ds4_gpu_matmul_f16_tensor(out_a, model_map, model_size,
                                    weight_a_offset, in_dim, out_dim,
                                    x, n_tok)) return 0;
    return ds4_gpu_matmul_f16_tensor(out_b, model_map, model_size,
                                     weight_b_offset, in_dim, out_dim,
                                     x, n_tok);
}
            
/* =========================================================================
 * ds4_gpu_compressor_prefill_state_ratio4_tensor
 *
 * Re-initializes the rolling compressor state of a ratio-4 layer from the
 * 4 projected tail rows (the last complete window of the prefill), exactly
 * like the CUDA call site in ds4_cuda.cu (compressor_set_rows_kernel with
 * src0=0, dst0=0, rows=4) and the Metal compressor_set_rows_projected path
 * (dst rows {0,1,2,3}) in ds4_metal.m.
 *
 * State layout (ratio-4): width = 2 * head_dim, state_rows = 8.
 *   rows [0,4)  : attention lane  -> filled from the tail window
 *   rows [4,8)  : indexer lane    -> left empty (kv 0 / score -INFINITY)
 * The APE scalar of phase (pos0 + src) % 4 is added to the score, exactly as
 * in compressor_set_rows / compressor_store_batch.  The engine calls this
 * after projecting the tail (ds4.c ~26336) with kv_tail/sc_tail = the 4-row
 * metal_graph_batch_comp_kv/sc buffers and pos0 = pos0 + n_tokens - 4.
 * ========================================================================= */
int ds4_gpu_compressor_prefill_state_ratio4_tensor(
        ds4_gpu_tensor       *state_kv,
        ds4_gpu_tensor       *state_score,
        const ds4_gpu_tensor *kv_tail,
        const ds4_gpu_tensor *sc_tail,
        const void             *model_map,
        uint64_t                model_size,
        uint64_t                ape_offset,
        uint32_t                ape_type,
        uint32_t                head_dim,
        uint32_t                pos0) {
    DS4_VK_TRACE_KERNEL("compressor_prefill_state_ratio4");
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) return 0;
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t ape_bytes = (uint64_t)ratio * width * (ape_type == 1u ? 2u : 4u);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) return 0;
    set_model_map_identity(model_map, model_size);
    VkBuffer ape_buf; VkDeviceSize ape_off, ape_range;
    if (!find_model_buffer(ape_offset, ape_bytes, ape_buf, ape_off, ape_range)) return 0;
    if (!compressor_clear_vk(state_kv, state_score, state_rows * width,
                             0.0f, -INFINITY)) return 0;
    return compressor_set_rows_vk(kv_tail, sc_tail, state_kv, state_score,
                                  ape_buf, ape_off, ape_type, width, ratio, pos0,
                                  0, 0, ratio);
#if 0
    DS4_VK_TRACE_KERNEL("compressor_prefill_state_ratio4");
    if (!state_kv || !state_score || !kv_tail || !sc_tail || !model_map ||
        head_dim == 0 || (ape_type != 0u && ape_type != 1u)) {
        return 0;
    }
    const uint32_t ratio = 4u;
    const uint32_t width = 2u * head_dim;
    const uint32_t state_rows = 8u;
    const uint64_t elem_ape = ape_type == 1u ? 2u : 4u;
    const uint64_t tail_bytes = (uint64_t)ratio * width * sizeof(float);
    const uint64_t state_bytes = (uint64_t)state_rows * width * sizeof(float);
    const uint64_t ape_bytes = (uint64_t)ratio * width * elem_ape;
    if (ape_offset > model_size || ape_bytes > model_size - ape_offset ||
        kv_tail->bytes < tail_bytes || sc_tail->bytes < tail_bytes ||
        state_kv->bytes < state_bytes || state_score->bytes < state_bytes) {
        return 0;
    }

    float *skvp = (float *)state_kv->ptr;
    float *sscp = (float *)state_score->ptr;
    const uint64_t state_n = (uint64_t)state_rows * width;
    memset(skvp, 0, (size_t)(state_n * sizeof(float)));
    for (uint64_t i = 0; i < state_n; i++) sscp[i] = -INFINITY;

    compressor_set_rows(skvp, sscp,
                        (const float *)kv_tail->ptr, (const float *)sc_tail->ptr,
                        model_map, ape_offset, ape_type,
                        width, ratio, pos0,
                        0, 0, ratio);
    return 1;
#endif
}

/* ---- AUTO-GENERATED CPU IMPLEMENTATIONS ---- */
extern "C" {
#include "_impl_gen.cpp"
}

/* ---- BEGIN AUTO-GENERATED STUBS ---- */
extern "C" {
#include "_stubs.gen.cpp"
}

extern "C" int ds4_gpu_build_derived_artifacts(
        const void *model_map, uint64_t model_size, const char *model_path) {
    (void)model_map;
    (void)model_size;
    (void)model_path;
    return 0;
}

extern "C" int ds4_gpu_model_range_replaced(
        const void *model_map, uint64_t offset, uint64_t bytes) {
    (void)model_map;
    (void)offset;
    (void)bytes;
    return 0;
}

extern "C" int ds4_gpu_matmul_f16_rms_fold_tensor(
        ds4_gpu_tensor       *out,
        const void           *model_map,
        uint64_t              model_size,
        uint64_t              weight_offset,
        uint64_t              in_dim,
        uint64_t              out_dim,
        const ds4_gpu_tensor *x,
        uint64_t              n_tok,
        float                 norm_eps) {
    (void)out;
    (void)model_map;
    (void)model_size;
    (void)weight_offset;
    (void)in_dim;
    (void)out_dim;
    (void)x;
    (void)n_tok;
    (void)norm_eps;
    return 0;
}

extern "C" int ds4_gpu_decode_graphs_supported(void) { return 0; }
extern "C" int ds4_gpu_decode_graph_begin(const ds4_decode_graph_key *key) {
    (void)key;
    return -1;
}
extern "C" int ds4_gpu_decode_graph_end(const ds4_decode_graph_key *key) {
    (void)key;
    return -1;
}
extern "C" void ds4_gpu_decode_graph_abort(const ds4_decode_graph_key *key) {
    (void)key;
}
extern "C" void ds4_gpu_decode_graphs_invalidate(void) {}

/* =====================================================================
 * Multi-GPU plumbing compatibility shims (single logical device).
 *
 * ds4.c and the CLI parser reference these on Linux builds.  The Vulkan
 * backend drives one logical device, so most of them are trivial.
 * ===================================================================== */

ds4_gpu_ctx g_gpu[DS4_MAX_GPUS] = {};
int g_n_gpus = 1;
int g_gpu_peer_ok[DS4_MAX_GPUS][DS4_MAX_GPUS] = {{1}};

extern "C" int ds4_gpu_init_multi(const ds4_gpu_config *cfg) {
    if (!cfg || cfg->n_gpus != 1) {
        fprintf(stderr, "ds4: Vulkan supports one GPU per process\n");
        return 0;
    }
    g_gpu[0].device_id = cfg->device_indices[0];
    g_n_gpus = 1;
    return ds4_gpu_init();
}

extern "C" int ds4_gpu_set_current_device(int logical_tier) {
    (void)logical_tier;
    return 0; /* single device: 0 = success */
}

extern "C" int ds4_gpu_set_current_device_fenced(int logical_tier) {
    return ds4_gpu_set_current_device(logical_tier);
}

extern "C" uint64_t ds4_gpu_tier_free_vram(int logical_tier) {
    if (logical_tier != 0) return 0;
    return g_vk.caps.device_memory_total;
}

extern "C" int ds4_gpu_tensor_alloc_on(ds4_gpu_tensor *t, int device_id,
                                       uint64_t bytes) {
    if (!t) return 1;
    if (device_id != 0) return 2;
    ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(bytes ? bytes : 1);
    if (!a) return 3;
    t->ptr = a->ptr; t->bytes = a->bytes; t->owner = a->owner;
    t->device_id = 0;
    free(a);
    return 0;
}

extern "C" void ds4_gpu_tensor_free_in_place(ds4_gpu_tensor *t) {
    if (!t) return;
    if (t->owner && t->ptr) {
        auto &ctx = get_cmd_ctx();
        if (ctx.layer_batch_active) {
            ctx.layer_batch_in_place_ptrs.push_back(t->ptr);
            memset(t, 0, sizeof(*t));
            return;
        }
        auto it = g_vk.tensor_headers.find(t->ptr);
        if (it != g_vk.tensor_headers.end()) {
            if (getenv("DS4_VULKAN_TIMELINE"))
                timeline_resource(get_cmd_ctx(), TimelineEventKind::TensorFree,
                                  "tensor", it->second->bytes);
            if (!release_tensor_header(it->second))
                g_vk.tensor_headers.erase(it);
        }
    }
    memset(t, 0, sizeof(*t));
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_ptr_on(int tier, uint64_t bytes) {
    if (tier != 0) return NULL;
    return ds4_gpu_tensor_alloc(bytes);
}

extern "C" ds4_gpu_tensor *ds4_gpu_tensor_alloc_managed_on(int tier, uint64_t bytes) {
    if (tier != 0) return NULL;
    return ds4_gpu_tensor_alloc_managed(bytes);
}

extern "C" int ds4_gpu_tensor_copy_async(ds4_gpu_tensor *dst,
                                          const ds4_gpu_tensor *src,
                                          uint64_t bytes) {
    return ds4_gpu_tensor_copy(dst, 0, src, 0, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev(ds4_gpu_tensor *dst,
                                         const ds4_gpu_tensor *src,
                                         uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev_default(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev_default(ds4_gpu_tensor *dst,
                                                 const ds4_gpu_tensor *src,
                                                 uint64_t bytes) {
    return ds4_gpu_tensor_copy(dst, 0, src, 0, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev_ordered(ds4_gpu_tensor *dst,
                                                 const ds4_gpu_tensor *src,
                                                 uint64_t bytes) {
    return ds4_gpu_tensor_copy_xdev_default(dst, src, bytes);
}

extern "C" int ds4_gpu_tensor_copy_xdev3(
        ds4_gpu_tensor       *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor       *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor       *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return (bytes0 == 0 || ds4_gpu_tensor_copy_xdev_default(dst0, src0, bytes0)) &&
           (bytes1 == 0 || ds4_gpu_tensor_copy_xdev_default(dst1, src1, bytes1)) &&
           (bytes2 == 0 || ds4_gpu_tensor_copy_xdev_default(dst2, src2, bytes2));
}

extern "C" int ds4_gpu_tensor_copy_xdev3_default_dst(
        ds4_gpu_tensor       *dst0, const ds4_gpu_tensor *src0, uint64_t bytes0,
        ds4_gpu_tensor       *dst1, const ds4_gpu_tensor *src1, uint64_t bytes1,
        ds4_gpu_tensor       *dst2, const ds4_gpu_tensor *src2, uint64_t bytes2) {
    return ds4_gpu_tensor_copy_xdev3(dst0, src0, bytes0, dst1, src1, bytes1,
                                     dst2, src2, bytes2);
}

extern "C" int ds4_gpu_tensor_wait_xdev(const ds4_gpu_tensor *src, int dst_tier) {
    return src && dst_tier == 0;
}

extern "C" int ds4_gpu_tensor_wait_xdev_default(const ds4_gpu_tensor *src,
                                                 int dst_tier) {
    return ds4_gpu_tensor_wait_xdev(src, dst_tier);
}

extern "C" int ds4_gpu_tensor_device(const ds4_gpu_tensor *t) {
    return t ? t->device_id : -1;
}

extern "C" int ds4_gpu_add_xdev_tensor(ds4_gpu_tensor *out,
                                        const ds4_gpu_tensor *local,
                                        const ds4_gpu_tensor *remote,
                                        ds4_gpu_tensor *remote_tmp,
                                        uint32_t n) {
    (void)remote_tmp;
    if (!out || !local || !remote || !out->ptr || !local->ptr || !remote->ptr)
        return 0;
    float *op = (float*)out->ptr;
    const float *lp = (const float*)local->ptr;
    const float *rp = (const float*)remote->ptr;
    for (uint32_t i = 0; i < n; i++) op[i] = lp[i] + rp[i];
    return 1;
}

extern "C" int ds4_gpu_register_model_map_no_copy(const void *model_map,
                                                   uint64_t model_size) {
    return ds4_gpu_set_model_map(model_map, model_size);
}

extern "C" int ds4_gpu_lookup_cache_strict(uint64_t source_offset,
                                            uint64_t bytes,
                                            int      expected_device,
                                            void   **out_device_ptr) {
    (void)source_offset; (void)bytes; (void)expected_device;
    if (out_device_ptr) *out_device_ptr = NULL;
    return 0;
}

extern "C" int ds4_gpu_lookup_cache(uint64_t source_offset, uint64_t bytes,
                                     int *out_device_id, void **out_device_ptr) {
    (void)source_offset; (void)bytes;
    if (out_device_id) *out_device_id = 0;
    if (out_device_ptr) *out_device_ptr = NULL;
    return 0;
}

extern "C" int ds4_gpu_lookup_cache_device(uint64_t source_offset, uint64_t bytes) {
    (void)source_offset; (void)bytes;
    return 0;
}

extern "C" int ds4_gpu_args_probe_auto_cuda(const int *device_filter,
                                             int filter_len,
                                             ds4_gpu_config *out,
                                             size_t safety_margin_bytes,
                                             char *errbuf,
                                             size_t errbuflen) {
    (void)device_filter; (void)filter_len; (void)out; (void)safety_margin_bytes;
    if (errbuf && errbuflen) {
        snprintf(errbuf, errbuflen,
                 "Vulkan: --gpu-vram auto is not supported; pass explicit budgets");
    }
    return 1;
}

extern "C" void ds4_gpu_enable_q8_dequant_gemm(void) {}

extern "C" int ds4_gpu_set_decode_fast_attention(int enabled) {
    (void)enabled; return 0;
}

extern "C" int ds4_gpu_set_decode_score_vec4(int enabled) {
    (void)enabled; return 0;
}

extern "C" int ds4_gpu_register_support_map(const void *map, uint64_t size,
                                             uint64_t bias) {
    (void)map; (void)size; (void)bias; return 0;
}

extern "C" int ds4_gpu_device_cache_tensors(int device_id,
                                             const ds4_tensor_range *ranges,
                                             int n_ranges) {
    (void)device_id; (void)ranges; (void)n_ranges; return 1;
}

extern "C" int ds4_gpu_device_cache_support_tensors(int device_id,
                                                     int entry_device_id,
                                                     const ds4_tensor_range *ranges,
                                                     int n_ranges,
                                                     int from_main_map) {
    (void)device_id; (void)entry_device_id; (void)ranges;
    (void)n_ranges; (void)from_main_map; return 1;
}

static int g_vk_q8_cache_suppressed = 0;

extern "C" int ds4_gpu_q8_cache_suppressed(void) {
    return g_vk_q8_cache_suppressed;
}

extern "C" void ds4_gpu_set_q8_cache_suppressed(int suppressed) {
    g_vk_q8_cache_suppressed = suppressed != 0;
}

static bool find_model_buffer(uint64_t offset, uint64_t bytes,
                              VkBuffer &buffer, VkDeviceSize &buffer_offset,
                              VkDeviceSize &range) {
    if (bytes == 0 || offset > g_vk.model_size || bytes > g_vk.model_size - offset)
        return false;
    if (!ensure_weight(offset, bytes)) return false;
    for (auto &[base, entry] : g_vk.weight_cache) {
        if (offset >= base && offset - base <= entry.size &&
            bytes <= entry.size - (offset - base)) {
            buffer = entry.buffer;
            buffer_offset = offset - base;
            range = bytes;
            entry.last_used = ++g_vk.lru_counter;
            mark_weight_generation(entry);
            timeline_resource(get_cmd_ctx(), TimelineEventKind::WeightUse,
                              "model_range", bytes, offset);
            return true;
        }
    }
    return false;
}
