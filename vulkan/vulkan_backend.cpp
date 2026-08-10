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
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <algorithm>

/* =====================================================================
 * PART 1: Vulkan Device State & Infrastructure
 * ===================================================================== */

struct VulkanCommandCtx {
    VkCommandPool   pool     = VK_NULL_HANDLE;
    VkCommandBuffer cmd      = VK_NULL_HANDLE;
    VkFence         fence    = VK_NULL_HANDLE;
    VkSemaphore     semaphore = VK_NULL_HANDLE;
    uint64_t        event_counter = 0;
    uint32_t        command_count = 0;
    bool            submitted = false;
    bool            recording = false;
    bool            first_cmd = true;
    VkDescriptorSet ds_q8s = VK_NULL_HANDLE;  /* simple shader DS */
    VkDescriptorSet ds_q8c = VK_NULL_HANDLE;  /* complex shader DS */
    VkCommandBuffer cmd_rots[4] = {};
    uint32_t cmd_rot_idx = 0;
    uint32_t cmd_buf_count = 0;
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
    uint64_t       bytes = 0;
    bool           is_managed = false;
};

/* ds4_gpu_tensor struct definition comes from ds4_gpu_mgpu.h (the header
 * forward-declares it in ds4_gpu.h).  The full layout
 * (ptr/bytes/owner/device_id) must match the shared multi-GPU plumbing. */
#include "../ds4_gpu_mgpu.h"

/* Forward declarations for VK_CHECK_RAW macro */
#define VK_CHECK_RAW(x) do { VkResult _r = (x); if (_r != VK_SUCCESS) { \
    fprintf(stderr, "ds4: VULKAN error %d at %s:%d\n", _r, __FILE__, __LINE__); return -1; } } while(0)
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
    
    std::mutex          cmd_mutex;
    std::unordered_map<std::thread::id, VulkanCommandCtx> cmd_ctxs;
    
    std::unordered_map<void*, TensorHeader*> tensor_headers;
    
    /* Weight cache: maps model file offset -> VkBuffer with weights copied to
     * GPU.  Ranges are uploaded lazily on first kernel use (see ensure_weight)
     * and evicted LRU-style against g_vk.weight_budget, so models larger than
     * the device heap stream layer-by-layer (llama.cpp-style). */
    struct WeightCacheEntry {
        VkBuffer       buffer = VK_NULL_HANDLE;
        VmaAllocation  allocation = VK_NULL_HANDLE;
        uint64_t       size = 0;
        uint64_t       last_used = 0;
        uint64_t       last_gen = 0;   /* command-buffer generation of last use */
        VkDescriptorBufferInfo desc_info{};
    };
    std::unordered_map<uint64_t, WeightCacheEntry> weight_cache;
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

    /* Check subgroup support */
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
    VkDescriptorSetLayoutBinding bindings[8] = {};
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
        {"matmul_q8_0_simple", 12, 6}, /* 3 x uint32: in_dim, out_dim, blocks */
        {"matmul_f16", 12, 6},   /* 3 x uint32 */
        {"rms_norm_weight_rows", 12, 6},
        {"head_rms_norm", 16, 6},  /* n_tok + n_head + head_dim + eps */
        {"rope_tail", 52, 6},      /* 7 x uint32 + 6 x float */
        {"head_rms_norm_rope_tail", 56, 6}, /* 7 x uint32 + 7 x float */
        {"store_raw_kv_f16", 16, 2},
        {"fp8_kv_quantize", 12, 1},
        {"attention_prefill_raw", 16, 4},
        {"attention_decode_mixed", 32, 6},
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
    std::lock_guard<std::mutex> lock(g_vk.cmd_mutex);
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
    VkCommandBufferAllocateInfo cbai{};
    cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cbai.commandPool = ctx.pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbai.commandBufferCount = 1;
    if (vkAllocateCommandBuffers(g_vk.device, &cbai, &ctx.cmd) != VK_SUCCESS) abort();
    VkFenceCreateInfo fci{}; fci.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    if (vkCreateFence(g_vk.device, &fci, nullptr, &ctx.fence) != VK_SUCCESS) abort();
    VkSemaphoreCreateInfo sci{}; sci.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    if (vkCreateSemaphore(g_vk.device, &sci, nullptr, &ctx.semaphore) != VK_SUCCESS) abort();
    g_vk.cmd_ctxs[tid] = ctx;
    return g_vk.cmd_ctxs[tid];
}

static int begin_cmd(void) {
    auto &c = get_cmd_ctx();
    if (c.recording) return 1;
    if (c.submitted) {
        VK_CHECK_RAW(vkWaitForFences(g_vk.device, 1, &c.fence, VK_TRUE, UINT64_MAX));
        VK_CHECK_RAW(vkResetFences(g_vk.device, 1, &c.fence));
        c.submitted = false;
        /* Pool cleanup every 4 submissions (llama.cpp: every 10) */
        c.cmd_buf_count++;
        if (c.cmd_buf_count >= 4) {
            /* llama.cpp-style cleanup.  RELEASE_RESOURCES also frees the
             * driver's internal command-stream buffers, which avoids a RADV
             * radv_amdgpu_cs_finalize crash after large prefill+decode. */
            vkResetCommandPool(g_vk.device, c.pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT);
            c.cmd_buf_count = 0;
            c.cmd_rot_idx = 0;
        } else {
            c.cmd_rot_idx = (c.cmd_rot_idx + 1) % 4;
        }
    }
    /* Allocate or reuse CB */
    VkCommandBuffer &cb = c.cmd_rots[c.cmd_rot_idx];
    if (cb == VK_NULL_HANDLE) {
        VkCommandBufferAllocateInfo cbai{};
        cbai.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        cbai.commandPool = c.pool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VK_CHECK_RAW(vkAllocateCommandBuffers(g_vk.device, &cbai, &cb));
    }
    c.cmd = cb;
    VkCommandBufferBeginInfo bi{}; bi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK_RAW(vkBeginCommandBuffer(c.cmd, &bi));
    c.recording = true;
    c.command_count = 0;
    g_vk.cmd_gen++;
    if (getenv("DS4_VULKAN_DEBUG"))
        fprintf(stderr, "ds4: [dbg] begin_cmd rot=%u gen=%llu\n",
                c.cmd_rot_idx, (unsigned long long)g_vk.cmd_gen);
    return 1;
}

#define DS4_VK_TRACE_KERNEL(name) \
    do { if (getenv("DS4_VULKAN_TRACE_KERNELS")) \
             fprintf(stderr, "ds4: [trace] %s\n", name); } while (0)

static int end_and_submit(void) {
    auto &c = get_cmd_ctx();
    if (!c.recording) return 1;
    if (c.command_count == 0) {
        VK_CHECK_RAW(vkEndCommandBuffer(c.cmd));
        c.recording = false;
        return 1;
    }
    if (getenv("DS4_VULKAN_DEBUG"))
        fprintf(stderr, "ds4: [dbg] end_and_submit cc=%u rot=%u gen=%llu\n",
                (unsigned)c.command_count, c.cmd_rot_idx, (unsigned long long)g_vk.cmd_gen);
    for (auto &[base, header] : g_vk.tensor_headers) {
        (void)base;
        (void)vmaFlushAllocation(g_vk.allocator, header->allocation, 0, header->bytes);
    }
    VK_CHECK_RAW(vkEndCommandBuffer(c.cmd));
    c.recording = false;
    VkSubmitInfo si{}; si.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.commandBufferCount = 1; si.pCommandBuffers = &c.cmd;
    VK_CHECK_RAW(vkQueueSubmit(g_vk.queue, 1, &si, c.fence));
    c.submitted = true;
    return 1;  /* DS4: non-zero = success */
}

static int wait_cmd(void) {
    auto &c = get_cmd_ctx();
    if (!c.submitted) return 1;
    VK_CHECK_RAW(vkWaitForFences(g_vk.device, 1, &c.fence, VK_TRUE, UINT64_MAX));
    for (auto &[base, header] : g_vk.tensor_headers) {
        (void)base;
        (void)vmaInvalidateAllocation(g_vk.allocator, header->allocation, 0, header->bytes);
    }
    return 1;  /* DS4: non-zero = success */
}

static int submit_and_wait(void) {
    int r = end_and_submit(); if (!r) return 0;
    return wait_cmd();
}

/* Split long command buffers into multiple submissions (llama.cpp-style):
 * RADV can crash finalizing a huge CS right after a large prefill, and
 * in-flight weight eviction is bounded by keeping command buffers short. */
static void maybe_submit(void) {
    auto &c = get_cmd_ctx();
    if (c.command_count >= 64) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr, "ds4: [dbg] maybe_submit cc=%u\n", (unsigned)c.command_count);
        end_and_submit();
        begin_cmd();
    }
}

/* ---- Compute Dispatch ---- */

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

    vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE,
                            e.layout, 0, 1, &ds, 0, nullptr);

    if (push && push_size)
        vkCmdPushConstants(c.cmd, e.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, push_size, push);

    vkCmdDispatch(c.cmd, gx, gy, gz);

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
    vkDeviceWaitIdle(g_vk.device);
    for (auto &[_, c] : g_vk.cmd_ctxs) {
        if (c.semaphore) vkDestroySemaphore(g_vk.device, c.semaphore, nullptr);
        if (c.fence) vkDestroyFence(g_vk.device, c.fence, nullptr);
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
        free(h);
    }
    g_vk.tensor_headers.clear();
    for (auto &[_, e] : g_vk.weight_cache)
        if (e.buffer) vmaDestroyBuffer(g_vk.allocator, e.buffer, e.allocation);
    g_vk.weight_cache.clear();
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

ds4_gpu_tensor *ds4_gpu_tensor_alloc(uint64_t bytes) {
    if (!bytes) return nullptr;
    ds4_gpu_tensor *t = (ds4_gpu_tensor*)calloc(1, sizeof(ds4_gpu_tensor));
    if (!t) return nullptr;

    VkBufferCreateInfo bci{};
    bci.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bci.size = bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    VmaAllocationCreateInfo aci{};
    aci.usage = VMA_MEMORY_USAGE_AUTO;
    aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VkBuffer buf; VmaAllocation alloc;
    VmaAllocationInfo ai;
    VkResult res = vmaCreateBuffer(g_vk.allocator, &bci, &aci, &buf, &alloc, &ai);
    if (res != VK_SUCCESS) {
        fprintf(stderr, "ds4: VULKAN tensor alloc failed for %lu bytes: VkResult=%d\n",
                (unsigned long)bytes, (int)res);
        free(t); return nullptr;
    }
    t->ptr = ai.pMappedData;
    t->bytes = bytes; t->owner = 1;

    TensorHeader *h = (TensorHeader*)calloc(1, sizeof(TensorHeader));
    h->buffer = buf; h->allocation = alloc; h->bytes = bytes;
    g_vk.tensor_headers[t->ptr] = h;
    return t;
}

static int ensure_weight(uint64_t offset, uint64_t needed_bytes);

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
    if (t->owner && t->ptr) {
        auto it = g_vk.tensor_headers.find(t->ptr);
        if (it != g_vk.tensor_headers.end()) {
            vmaDestroyBuffer(g_vk.allocator, it->second->buffer, it->second->allocation);
            free(it->second);
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
int ds4_gpu_commands_active(void) { return get_cmd_ctx().recording ? 1 : 0; }
int ds4_gpu_flush_commands(void) {
    /* The engine keeps recording after a flush (e.g. SSD streaming async
     * loads), so start a fresh command buffer like Metal's next encoder. */
    int ok = end_and_submit();
    if (ok) ok = begin_cmd();
    return ok;
}
int ds4_gpu_end_commands(void) { return submit_and_wait(); }
int ds4_gpu_synchronize(void) { VK_CHECK_RAW(vkDeviceWaitIdle(g_vk.device)); return 1; }

int ds4_gpu_signal_selected_readback_ready(uint64_t *ev) {
    auto &c = get_cmd_ctx(); *ev = ++c.event_counter; return 1;
}

int ds4_gpu_commit_and_wait_selected_readback(uint64_t ev, const char *label) {
    (void)ev; (void)label;
    /* End + wait + re-begin: the engine reads the selected ids on the CPU and
     * then keeps encoding GPU kernels (routed MoE) without a begin_commands. */
    int ok = end_and_submit();
    if (ok) ok = wait_cmd();
    if (ok) ok = begin_cmd();
    return ok;
}

int ds4_gpu_wait_selected_readback_ready(uint64_t ev, const char *label) {
    (void)ev; (void)label; return wait_cmd();
}

/* ---- Model Loading ---- */

static void clear_weight_cache(void) {
    if (!g_vk.weight_cache.empty()) {
        (void)vkDeviceWaitIdle(g_vk.device);
        for (auto &[offset, entry] : g_vk.weight_cache) {
            (void)offset;
            vmaDestroyBuffer(g_vk.allocator, entry.buffer, entry.allocation);
        }
        g_vk.weight_cache.clear();
        g_vk.weight_used = 0;
    }
    g_vk.range_registry.clear();
}

static void set_model_map_identity(const void *model_map, uint64_t model_size) {
    if (g_vk.model_map != model_map) clear_weight_cache();
    g_vk.model_map = model_map;
    g_vk.model_size = model_size;
}

int ds4_gpu_set_model_map(const void *m, uint64_t s) {
    if (!m) return 0;
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

/* Model ranges are only registered here (metadata).  The actual GPU upload
 * happens lazily in the kernels via ensure_weight(), so models larger than
 * the device heap stream layer-by-layer (llama.cpp-style) instead of failing
 * during startup. */
int ds4_gpu_cache_model_range(const void *m, uint64_t s, uint64_t off, uint64_t bytes, const char *label) {
    (void)m; (void)s; (void)label;
    if (bytes == 0) return 0;
    g_vk.range_registry[off] = bytes;
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
            e.last_gen = g_vk.cmd_gen;
            return 1;
        }
    }
    if (!g_vk.model_map || needed_bytes == 0) return 0;

    uint64_t size = needed_bytes;
    for (auto &[rb, rs] : g_vk.range_registry) {
        if (offset >= rb && offset < rb + rs) { size = rs > needed_bytes ? rs : needed_bytes; break; }
    }
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
                        if (vkQueueSubmit(g_vk.queue, 1, &si, fence) == VK_SUCCESS) {
                            vkWaitForFences(g_vk.device, 1, &fence, VK_TRUE, UINT64_MAX);
                            staged = true;
                        }
                        vkDestroyFence(g_vk.device, fence, nullptr);
                    }
                }
                vkFreeCommandBuffers(g_vk.device, load_pool, 1, &cb);
            }
        }
        vmaDestroyBuffer(g_vk.allocator, sbuf, salloc);
    }
    if (!staged) { vmaDestroyBuffer(g_vk.allocator, buf, alloc); return 0; }

    g_vk.weight_cache[offset] = {buf, alloc, size, ++g_vk.lru_counter, g_vk.cmd_gen, {}};
    g_vk.weight_used += size;

    /* LRU eviction (never evict the range just uploaded, nor any range still
     * referenced by the command buffer currently being recorded). */
    while (g_vk.weight_used > g_vk.weight_budget) {
        uint64_t lru_base = UINT64_MAX, lru_time = UINT64_MAX;
        for (auto &[b, e] : g_vk.weight_cache) {
            if (b == offset) continue;
            if (e.last_gen == g_vk.cmd_gen) continue;
            if (e.last_used < lru_time) { lru_time = e.last_used; lru_base = b; }
        }
        if (lru_base == UINT64_MAX) break;
        auto it = g_vk.weight_cache.find(lru_base);
        vmaDestroyBuffer(g_vk.allocator, it->second.buffer, it->second.allocation);
        g_vk.weight_used -= it->second.size;
        g_vk.weight_cache.erase(it);
    }
    return 1;
}

int ds4_gpu_cache_q8_f16_range(const void *m, uint64_t s, uint64_t off, uint64_t bytes,
                                 uint64_t idim, uint64_t odim, const char *label) {
    (void)m; (void)s; (void)off; (void)bytes; (void)idim; (void)odim; (void)label; return 0; }

void ds4_gpu_release_q8_f16_cache(void) {}
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

static bool shader_f32_domain(uint64_t a, uint64_t b, uint64_t c) {
    if (a != 0 && b > UINT32_MAX / a) return false;
    const uint64_t ab = a * b;
    return ab == 0 || c <= UINT32_MAX / ab;
}

static int finish_simple_dispatch(VulkanCommandCtx &ctx, bool resume_recording) {
    VkMemoryBarrier barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(ctx.cmd,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &barrier, 0, nullptr, 0, nullptr);
    ctx.command_count++;
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
    return 1;
}

static int release_simple_descriptors(VkDescriptorSet set) {
    return vkFreeDescriptorSets(g_vk.device, g_vk.desc_pool, 1, &set) == VK_SUCCESS;
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
    vkCmdDispatch(ctx.cmd, gx, gy, gz);
    ctx.command_count++;
    int ok = finish_simple_dispatch(ctx, resume_recording);
    if (!release_simple_descriptors(set)) ok = 0;
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
    vkCmdDispatch(ctx.cmd, rows, 1, 1);
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
    vkCmdDispatch(ctx.cmd, (n + 255) / 256, 1, 1);
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
    vkCmdDispatch(ctx.cmd, (n + 255) / 256, 1, 1);
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
        wit->second.last_gen = g_vk.cmd_gen;
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
        vkCmdBindPipeline(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sh.pipeline);
        vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sh.layout, 0, 1, &ds, 0, nullptr);
        const uint32_t y_scale = std::min((uint32_t)out_dim, 65534u);
        const uint32_t y_cnt = ((uint32_t)out_dim + y_scale - 1) / y_scale;
        struct { uint32_t in_dim, out_dim, n_tok, blocks, y_scale; } pc = {
            (uint32_t)in_dim, (uint32_t)out_dim, tile_n, (uint32_t)n_blocks, y_scale
        };
        vkCmdPushConstants(c.cmd, sh.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(c.cmd, y_scale, y_cnt, tile_n);
        c.command_count++;
        VkMemoryBarrier mb{};
        mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
        mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb,
                             0, nullptr, 0, nullptr);
        if (!submit_and_wait()) return 0;
        if (vkFreeDescriptorSets(g_vk.device, g_vk.desc_pool, 1, &ds) != VK_SUCCESS) return 0;
    }
    return 1;
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
    (void)model_map; (void)model_size;
    if (!out || !x) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr, "ds4: [dbg] matmul_f16 missing tensor\n");
        return 0;
    }

    auto si = g_vk.shader_map.find("matmul_f16");
    if (si == g_vk.shader_map.end()) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr, "ds4: [dbg] matmul_f16 shader unavailable\n");
        return 0;
    }
    auto &sh = g_vk.shaders[si->second];
    auto &c = get_cmd_ctx();
    if (!c.recording && !begin_cmd()) return 0;
    vkCmdBindPipeline(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sh.pipeline);

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
    VkDescriptorSetAllocateInfo dai{};
    dai.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    dai.descriptorPool = g_vk.desc_pool; dai.descriptorSetCount = 1;
    dai.pSetLayouts = &sh.desc_layout;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    const VkResult alloc_result = vkAllocateDescriptorSets(g_vk.device, &dai, &ds);
    if (alloc_result != VK_SUCCESS) {
        if (getenv("DS4_VULKAN_DEBUG"))
            fprintf(stderr,
                    "ds4: [dbg] matmul_f16 descriptor allocation failed: %d\n",
                    alloc_result);
        return 0;
    }
    VkDeviceSize x_size = std::min<VkDeviceSize>(xbuf == obuf ? (ooff - xoff) : VK_WHOLE_SIZE, in_dim * n_tok * sizeof(float));
    const VkDeviceSize w_size = std::min<VkDeviceSize>(
        wit->second.size - (wbuf_off > wit->second.size ? 0 : wbuf_off),
        weight_bytes);  /* f16 weights: 2 bytes per element */
    VkDeviceSize o_size = out_dim * n_tok * sizeof(float);
    VkDescriptorBufferInfo bufs[3] = {
        {xbuf, xoff, x_size}, {wbuf, wbuf_off, w_size}, {obuf, ooff, o_size},
    };
    VkWriteDescriptorSet w[3];
    for (int i = 0; i < 3; i++) {
        w[i] = {}; w[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        w[i].dstSet = ds; w[i].dstBinding = i; w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; w[i].pBufferInfo = &bufs[i];
    }
    vkUpdateDescriptorSets(g_vk.device, 3, w, 0, nullptr);
    vkCmdBindDescriptorSets(c.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sh.layout, 0, 1, &ds, 0, nullptr);

    struct { uint32_t in_dim, out_dim, n_tok; } pc = {
        (uint32_t)in_dim, (uint32_t)out_dim, (uint32_t)n_tok
    };
    vkCmdPushConstants(c.cmd, sh.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
    if ((uint32_t)out_dim <= 65534) {
        vkCmdDispatch(c.cmd, (uint32_t)out_dim, (uint32_t)n_tok, 1);
    } else {
        const uint32_t max_wg = 65534;
        uint32_t dispatched = 0;
        while (dispatched < (uint32_t)out_dim) {
            uint32_t chunk = std::min((uint32_t)out_dim - dispatched, max_wg);
            vkCmdDispatch(c.cmd, chunk, (uint32_t)n_tok, 1);
            dispatched += chunk;
        }
    }
    c.command_count++;

    /* Memory barrier: visibility for subsequent dispatches */
    VkMemoryBarrier mb{};
    mb.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(c.cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);

    maybe_submit();
    int ok = submit_and_wait();
    if (ok && vkFreeDescriptorSets(g_vk.device, g_vk.desc_pool, 1, &ds) != VK_SUCCESS) ok = 0;
    return ok;
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
    wit->second.last_gen = g_vk.cmd_gen;
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
    vkCmdDispatch(ctx.cmd, rows, 1, 1);
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
        struct Push {
            uint32_t n_tok, n_head, head_dim, n_rot, pos0, n_ctx_orig;
            int32_t inverse;
            float freq_base, freq_scale, ext_factor, attn_factor, beta_fast, beta_slow, eps;
        } push = {tile_n, n_head, head_dim, n_rot, pos0 + token_base, n_ctx_orig,
                  inverse ? 1 : 0, freq_base, freq_scale, ext_factor, attn_factor,
                  beta_fast, beta_slow, eps};
        vkCmdPushConstants(ctx.cmd, shader.layout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(push), &push);
        vkCmdDispatch(ctx.cmd, tile_n * n_head, 1, 1);
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
        vkCmdDispatch(ctx.cmd, tile_n * n_head, 1, 1);
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
    const uint32_t tile_tokens = (uint32_t)std::min<uint64_t>(n_tok,
        (65535ull * 256ull) / pairs_per_token);
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
        const uint64_t tile_pairs = (uint64_t)tile_n * pairs_per_token;
        const uint32_t groups = (uint32_t)((tile_pairs + 255u) / 256u);
        if (groups == 0 || groups > 65535u) {
            if (!release_simple_descriptors(set)) return 0;
            return fail_simple_dispatch(ctx);
        }
        vkCmdDispatch(ctx.cmd, groups, 1, 1);
        ok = finish_simple_dispatch(ctx, resume_recording);
        if (!release_simple_descriptors(set)) ok = 0;
        if (!ok) return 0;
        token_base += tile_n;
    }
    return ok;
}

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
    VkDescriptorBufferInfo bufs[6] = {
        {obuf, ooff, (VkDeviceSize)heads->bytes}, {qbuf, qoff, (VkDeviceSize)q->bytes},
        {rbuf, roff, (VkDeviceSize)raw_kv->bytes},
        {cbuf, coff, n_comp ? (VkDeviceSize)comp_kv->bytes : 4},
        {mbuf, moff, use_mask ? (VkDeviceSize)comp_mask->bytes : 4}, {sbuf, soff, ssize}
    };
    struct { uint32_t n_raw, raw_cap, raw_start, n_comp, comp_f16, use_mask, n_head, head_dim; }
        pc = {n_raw, raw_cap, raw_start, n_comp, comp_kv_f16, use_mask, n_head, head_dim};
    DS4_VK_TRACE_KERNEL("attention_decode_mixed");
    return record_simple_shader("attention_decode_mixed", &pc, sizeof(pc), bufs, 6,
                                n_head, 1, 1, resume_recording);
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
 * Exact math replicated from ds4.c layer_grouped_out_batch /
 * matvec_q8_0_grouped_rows / matmul_q8_0_batch:
 *   1. Quantize every group slice of the input activation to Q8_0
 *      (scale = amax/127, q = clamp(lrintf(x*127/amax)), tail lanes 0).
 *   2. low[g*rank + r] = dot_q8_0_row(out_a row g*rank + r, xq_g, scale_g)
 *   3. Quantize the whole low vector, then
 *      out[o] = dot_q8_0_row(out_b row o, xq_low, scale_low)
 * Q8_0 rows are GGUF blocks of {f16 scale, 32 x int8} = 34 bytes per block.
 * The activation-side quantization is mandatory: the reference dot kernels
 * consume pre-quantized Q8_0 activations, not raw floats.
 */

/* Q8_0 activation quantization (identical to ds4.c quantize_q8_0_activation). */
static void ds4_attn_out_quant_q8_0(const float *x, int8_t *xq, float *scale, uint64_t n) {
    const uint64_t blocks = (n + 31u) / 32u;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = n - i0 < 32u ? n - i0 : 32u;
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
        for (uint64_t i = bn; i < 32u && i0 + i < blocks * 32u; i++) xq[i0 + i] = 0;
    }
}

/* Dot one Q8_0 weight row against a pre-quantized Q8_0 activation
 * (identical to ds4.c dot_q8_0_row). */
static float ds4_attn_out_dot_q8_0(const uint8_t *row, const int8_t *xq,
                                   const float *xscale, uint64_t in_dim,
                                   uint64_t blocks) {
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        uint16_t scale_bits;
        memcpy(&scale_bits, row + b * 34u, sizeof(scale_bits));
        const int8_t *qs = (const int8_t *)(row + b * 34u + 2u);
        const uint64_t i0 = b * 32u;
        const uint64_t n = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        int32_t sum = 0;
        for (uint64_t i = 0; i < n; i++) sum += (int32_t)qs[i] * (int32_t)xq[i0 + i];
        acc += ds4_half_to_float(scale_bits) * xscale[b] * (float)sum;
    }
    return acc;
}

int ds4_gpu_attention_output_low_q8_tensor(ds4_gpu_tensor *low,
    const void *model_map, uint64_t model_size, uint64_t out_a_offset,
    uint64_t group_dim, uint64_t rank, uint32_t n_groups,
    const ds4_gpu_tensor *heads)
{
    if (!low || !heads || !model_map || group_dim == 0 || rank == 0 || n_groups == 0)
        return 0;
    if (!low->ptr || !heads->ptr) return 0;

    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t out_a_bytes = low_dim * row_a_bytes;

    /* Safety: never read past the model mmap, never write past tensor bytes. */
    if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset) return 0;
    if ((uint64_t)n_groups * group_dim * sizeof(float) > heads->bytes) return 0;
    if (low_dim * sizeof(float) > low->bytes) return 0;

    const uint8_t *wa = (const uint8_t *)model_map + out_a_offset;
    const float *hp = (const float *)heads->ptr;
    float *lp = (float *)low->ptr;

    /* Quantize each group's activation slice once (n_groups * blocks). */
    std::vector<int8_t> xq((size_t)n_groups * blocks_a * 32u);
    std::vector<float> xscale((size_t)n_groups * blocks_a);
    for (uint32_t g = 0; g < n_groups; g++) {
        ds4_attn_out_quant_q8_0(hp + (uint64_t)g * group_dim,
                                xq.data() + (size_t)g * blocks_a * 32u,
                                xscale.data() + (size_t)g * blocks_a,
                                group_dim);
    }
    /* tensor_row == g*rank + r == idx (ds4.c matvec_q8_0_grouped_worker). */
    for (uint64_t idx = 0; idx < low_dim; idx++) {
        const uint64_t g = idx / rank;
        lp[idx] = ds4_attn_out_dot_q8_0(wa + idx * row_a_bytes,
                                        xq.data() + (size_t)g * blocks_a * 32u,
                                        xscale.data() + (size_t)g * blocks_a,
                                        group_dim, blocks_a);
    }
    return 1;
}

int ds4_gpu_attention_output_q8_batch_tensor(ds4_gpu_tensor *out, ds4_gpu_tensor *low,
    ds4_gpu_tensor *gt, ds4_gpu_tensor *lt, const void *mm, uint64_t ms,
    uint64_t oa_off, uint64_t ob_off, uint64_t gd, uint64_t rank,
    uint32_t ng, uint64_t od, const ds4_gpu_tensor *heads, uint32_t nt)
{
    (void)gt; (void)lt;
    if (!out || !low || !heads || !mm || gd == 0 || rank == 0 ||
        ng == 0 || od == 0 || nt == 0) return 0;
    if (!out->ptr || !low->ptr || !heads->ptr) return 0;

    const uint64_t low_dim = (uint64_t)ng * rank;
    const uint64_t blocks_a = (gd + 31u) / 32u;
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t out_a_bytes = low_dim * row_a_bytes;
    const uint64_t blocks_b = (low_dim + 31u) / 32u;
    const uint64_t row_b_bytes = blocks_b * 34u;
    const uint64_t out_b_bytes = od * row_b_bytes;

    /* Safety: model ranges and tensor byte sizes. */
    if (oa_off > ms || out_a_bytes > ms - oa_off) return 0;
    if (ob_off > ms || out_b_bytes > ms - ob_off) return 0;
    if ((uint64_t)nt * ng * gd * sizeof(float) > heads->bytes) return 0;
    if ((uint64_t)nt * low_dim * sizeof(float) > low->bytes) return 0;
    if ((uint64_t)nt * od * sizeof(float) > out->bytes) return 0;

    const uint8_t *wa = (const uint8_t *)mm + oa_off;
    const uint8_t *wb = (const uint8_t *)mm + ob_off;
    const float *hp = (const float *)heads->ptr;
    float *lp = (float *)low->ptr;
    float *op = (float *)out->ptr;

    /* Stage A: quantize every (token, group) activation slice once. */
    std::vector<int8_t> axq((size_t)nt * ng * blocks_a * 32u);
    std::vector<float> axscale((size_t)nt * ng * blocks_a);
    for (uint32_t t = 0; t < nt; t++) {
        const float *ht = hp + (uint64_t)t * ng * gd;
        for (uint32_t g = 0; g < ng; g++) {
            const size_t base = ((size_t)t * ng + g) * blocks_a;
            ds4_attn_out_quant_q8_0(ht + (uint64_t)g * gd,
                                    axq.data() + base * 32u,
                                    axscale.data() + base,
                                    gd);
        }
    }
    /* low[t][g*rank + r] = dot(out_a row g*rank + r, q8(heads[t][g])). */
    for (uint32_t t = 0; t < nt; t++) {
        float *ltp = lp + (uint64_t)t * low_dim;
        for (uint64_t idx = 0; idx < low_dim; idx++) {
            const uint64_t g = idx / rank;
            const size_t base = ((size_t)t * ng + (uint32_t)g) * blocks_a;
            ltp[idx] = ds4_attn_out_dot_q8_0(wa + idx * row_a_bytes,
                                             axq.data() + base * 32u,
                                             axscale.data() + base,
                                             gd, blocks_a);
        }
    }

    /* Stage B: quantize low per token, out[t][o] = dot(out_b row o, q8(low[t])). */
    std::vector<int8_t> bxq((size_t)nt * blocks_b * 32u);
    std::vector<float> bxscale((size_t)nt * blocks_b);
    for (uint32_t t = 0; t < nt; t++) {
        ds4_attn_out_quant_q8_0(lp + (uint64_t)t * low_dim,
                                bxq.data() + (size_t)t * blocks_b * 32u,
                                bxscale.data() + (size_t)t * blocks_b,
                                low_dim);
    }
    for (uint64_t o = 0; o < od; o++) {
        const uint8_t *row = wb + o * row_b_bytes;
        for (uint32_t t = 0; t < nt; t++) {
            op[(uint64_t)t * od + o] =
                ds4_attn_out_dot_q8_0(row,
                                      bxq.data() + (size_t)t * blocks_b * 32u,
                                      bxscale.data() + (size_t)t * blocks_b,
                                      low_dim, blocks_b);
        }
    }
    return 1;
}

/* ---- ds4_gpu_attention_output_q8_batch_f16_tensor ----
 *
 * Identical math to ds4_gpu_attention_output_q8_batch_tensor above
 * (grouped Q8_0 attention output projection), but the final out_dim-wide
 * projection of every token is stored as IEEE f16 (2 bytes/element) into
 * out_h instead of f32.  The engine's fast prefill path consumes
 * g->batch_q_half directly, so each output is rounded with
 * ds4_float_to_half (the same half rounding the engine applies elsewhere).
 * low stays f32 (the stage-A low vector buffer is shared with the f32
 * variant).
 */
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
    DS4_VK_TRACE_KERNEL("attention_output_q8_batch_f16");
    if (!out_h || !low || !heads || !model_map || group_dim == 0 || rank == 0 ||
        n_groups == 0 || out_dim == 0 || n_tokens == 0) return 0;
    if (!out_h->ptr || !low->ptr || !heads->ptr) return 0;

    const uint64_t low_dim = (uint64_t)n_groups * rank;
    const uint64_t blocks_a = (group_dim + 31u) / 32u;
    const uint64_t row_a_bytes = blocks_a * 34u;
    const uint64_t out_a_bytes = low_dim * row_a_bytes;
    const uint64_t blocks_b = (low_dim + 31u) / 32u;
    const uint64_t row_b_bytes = blocks_b * 34u;
    const uint64_t out_b_bytes = out_dim * row_b_bytes;

    /* Safety: model ranges and tensor byte sizes (out_h is f16!). */
    if (out_a_offset > model_size || out_a_bytes > model_size - out_a_offset) return 0;
    if (out_b_offset > model_size || out_b_bytes > model_size - out_b_offset) return 0;
    if ((uint64_t)n_tokens * n_groups * group_dim * sizeof(float) > heads->bytes) return 0;
    if ((uint64_t)n_tokens * low_dim * sizeof(float) > low->bytes) return 0;
    if ((uint64_t)n_tokens * out_dim * sizeof(uint16_t) > out_h->bytes) return 0;

    const uint8_t *wa = (const uint8_t *)model_map + out_a_offset;
    const uint8_t *wb = (const uint8_t *)model_map + out_b_offset;
    const float *hp = (const float *)heads->ptr;
    float *lp = (float *)low->ptr;
    uint16_t *op = (uint16_t *)out_h->ptr;

    /* Stage A: quantize every (token, group) activation slice once. */
    std::vector<int8_t> axq((size_t)n_tokens * n_groups * blocks_a * 32u);
    std::vector<float> axscale((size_t)n_tokens * n_groups * blocks_a);
    for (uint32_t t = 0; t < n_tokens; t++) {
        const float *ht = hp + (uint64_t)t * n_groups * group_dim;
        for (uint32_t g = 0; g < n_groups; g++) {
            const size_t base = ((size_t)t * n_groups + g) * blocks_a;
            ds4_attn_out_quant_q8_0(ht + (uint64_t)g * group_dim,
                                    axq.data() + base * 32u,
                                    axscale.data() + base,
                                    group_dim);
        }
    }
    /* low[t][g*rank + r] = dot(out_a row g*rank + r, q8(heads[t][g])). */
    for (uint32_t t = 0; t < n_tokens; t++) {
        float *ltp = lp + (uint64_t)t * low_dim;
        for (uint64_t idx = 0; idx < low_dim; idx++) {
            const uint64_t g = idx / rank;
            const size_t base = ((size_t)t * n_groups + (uint32_t)g) * blocks_a;
            ltp[idx] = ds4_attn_out_dot_q8_0(wa + idx * row_a_bytes,
                                             axq.data() + base * 32u,
                                             axscale.data() + base,
                                             group_dim, blocks_a);
        }
    }

    /* Stage B: quantize low per token, out_h[t][o] =
     * half(dot(out_b row o, q8(low[t]))). */
    std::vector<int8_t> bxq((size_t)n_tokens * blocks_b * 32u);
    std::vector<float> bxscale((size_t)n_tokens * blocks_b);
    for (uint32_t t = 0; t < n_tokens; t++) {
        ds4_attn_out_quant_q8_0(lp + (uint64_t)t * low_dim,
                                bxq.data() + (size_t)t * blocks_b * 32u,
                                bxscale.data() + (size_t)t * blocks_b,
                                low_dim);
    }
    for (uint64_t o = 0; o < out_dim; o++) {
        const uint8_t *row = wb + o * row_b_bytes;
        for (uint32_t t = 0; t < n_tokens; t++) {
            const float v = ds4_attn_out_dot_q8_0(row,
                                                  bxq.data() + (size_t)t * blocks_b * 32u,
                                                  bxscale.data() + (size_t)t * blocks_b,
                                                  low_dim, blocks_b);
            op[(uint64_t)t * out_dim + o] = ds4_float_to_half(v);
        }
    }
    return 1;
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
    DS4_VK_TRACE_KERNEL("attention_prefill_static_mixed");
    if (!heads || !heads->ptr || !q || !q->ptr || !raw_kv || !raw_kv->ptr ||
        !model_map || n_tokens == 0 || n_head == 0 || head_dim == 0 ||
        (n_comp != 0 && (!comp_kv || !comp_kv->ptr)))
        return 0;
    const uint64_t head_bytes = (uint64_t)n_tokens * n_head * head_dim * sizeof(float);
    const uint64_t kv_bytes = (uint64_t)n_tokens * head_dim * sizeof(float);
    const uint64_t comp_elem = comp_kv_f16 ? sizeof(uint16_t) : sizeof(float);
    const uint64_t sink_bytes = (uint64_t)n_head * sizeof(float);
    if (sinks_offset > model_size || sink_bytes > model_size - sinks_offset ||
        heads->bytes < head_bytes || q->bytes < head_bytes ||
        raw_kv->bytes < kv_bytes ||
        (n_comp != 0 && comp_kv->bytes < (uint64_t)n_comp * head_dim * comp_elem))
        return 0;

    const float *sinks = (const float *)((const char *)model_map + sinks_offset);
    const float *qp = (const float *)q->ptr;
    const float *rawp = (const float *)raw_kv->ptr;
    const uint8_t *compp = (n_comp && comp_kv) ? (const uint8_t *)comp_kv->ptr : nullptr;
    float *hp = (float *)heads->ptr;
    const float scale = 1.0f / sqrtf((float)head_dim);

    std::vector<float> score;
    for (uint32_t t = 0; t < n_tokens; t++) {
        const uint32_t raw_count = (window != 0 && t + 1u > window) ? window : (t + 1u);
        const uint32_t raw_start = t + 1u - raw_count;
        uint32_t comp_count = 0;
        if (n_comp != 0 && ratio != 0) {
            comp_count = (t + 1u) / ratio;
            if (comp_count > n_comp) comp_count = n_comp;
        }
        const uint32_t n_total = raw_count + comp_count;
        score.resize(n_total);

        for (uint32_t h = 0; h < n_head; h++) {
            const float *qh = qp + ((uint64_t)t * n_head + h) * head_dim;
            float max_score = sinks[h];

            for (uint32_t r = 0; r < raw_count; r++) {
                const float *kv = rawp + (uint64_t)(raw_start + r) * head_dim;
                float dot = 0.0f;
                for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kv[d];
                score[r] = dot * scale;
                if (score[r] > max_score) max_score = score[r];
            }
            for (uint32_t c = 0; c < comp_count; c++) {
                const uint32_t idx = raw_count + c;
                const uint8_t *kv = compp + (uint64_t)c * head_dim * comp_elem;
                float dot = 0.0f;
                if (comp_kv_f16) {
                    const uint16_t *kvh = (const uint16_t *)kv;
                    for (uint32_t d = 0; d < head_dim; d++)
                        dot += qh[d] * ds4_half_to_float(kvh[d]);
                } else {
                    const float *kvf = (const float *)kv;
                    for (uint32_t d = 0; d < head_dim; d++) dot += qh[d] * kvf[d];
                }
                score[idx] = dot * scale;
                if (score[idx] > max_score) max_score = score[idx];
            }

            float *oh = hp + ((uint64_t)t * n_head + h) * head_dim;
            std::memset(oh, 0, (size_t)head_dim * sizeof(oh[0]));
            float denom = expf(sinks[h] - max_score);
            for (uint32_t r = 0; r < raw_count; r++) {
                const float w = expf(score[r] - max_score);
                const float *kv = rawp + (uint64_t)(raw_start + r) * head_dim;
                denom += w;
                for (uint32_t d = 0; d < head_dim; d++) oh[d] += w * kv[d];
            }
            for (uint32_t c = 0; c < comp_count; c++) {
                const uint32_t idx = raw_count + c;
                const uint8_t *kv = compp + (uint64_t)c * head_dim * comp_elem;
                const float w = expf(score[idx] - max_score);
                denom += w;
                if (comp_kv_f16) {
                    const uint16_t *kvh = (const uint16_t *)kv;
                    for (uint32_t d = 0; d < head_dim; d++)
                        oh[d] += w * ds4_half_to_float(kvh[d]);
                } else {
                    const float *kvf = (const float *)kv;
                    for (uint32_t d = 0; d < head_dim; d++) oh[d] += w * kvf[d];
                }
            }
            const float inv = 1.0f / denom;
            for (uint32_t d = 0; d < head_dim; d++) oh[d] *= inv;
        }
    }
    return 1;
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
    vkCmdDispatch(ctx.cmd, (n_embd + 255u) / 256u, rows, 1);
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
    const uint32_t comb_stride = post_stride;
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
    vkCmdDispatch(ctx.cmd, (n_embd + 255u) / 256u, rows * n_hc, 1);
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
            it->second.last_gen = g_vk.cmd_gen;
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
    vkCmdDispatch(ctx.cmd, 1, (uint32_t)rows, 1);
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
    vkCmdDispatch(ctx.cmd, (n_hc + 255u) / 256u, n_tokens, 1);
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

static float ds4_router_probability(float logit) {
    float softplus;
    if (logit > 20.0f) softplus = logit;
    else if (logit < -20.0f) softplus = expf(logit);
    else softplus = log1pf(expf(logit));
    return sqrtf(softplus);
}

static int ds4_router_select_row(
        int32_t       *selected,
        float         *weights,
        float         *probs,
        const float   *logits,
        const float   *bias,
        const int32_t *hash,
        uint32_t       hash_rows,
        int32_t        token,
        uint32_t       n_expert,
        uint32_t       n_expert_used,
        float          expert_weight_scale,
        bool           hash_mode) {
    for (uint32_t i = 0; i < n_expert; i++) {
        probs[i] = ds4_router_probability(logits[i]);
    }

    if (hash_mode) {
        if (!hash || hash_rows == 0) return 0;
        if (token < 0 || (uint32_t)token >= hash_rows) token = 0;
        const int32_t *row = hash + (uint64_t)(uint32_t)token * n_expert_used;
        for (uint32_t i = 0; i < n_expert_used; i++) selected[i] = row[i];
    } else {
        for (uint32_t i = 0; i < n_expert_used; i++) selected[i] = -1;
        for (uint32_t expert = 0; expert < n_expert; expert++) {
            const float score = probs[expert] + (bias ? bias[expert] : 0.0f);
            for (uint32_t rank = 0; rank < n_expert_used; rank++) {
                const int32_t current = selected[rank];
                const float current_score = current >= 0
                    ? probs[(uint32_t)current] + (bias ? bias[(uint32_t)current] : 0.0f)
                    : -INFINITY;
                if (current < 0 || score > current_score ||
                    (score == current_score && expert < (uint32_t)current)) {
                    for (uint32_t tail = n_expert_used - 1; tail > rank; tail--)
                        selected[tail] = selected[tail - 1];
                    selected[rank] = (int32_t)expert;
                    break;
                }
            }
        }
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < n_expert_used; i++) {
        const int32_t expert = selected[i];
        const float value = expert >= 0 && (uint32_t)expert < n_expert
            ? probs[(uint32_t)expert] : 0.0f;
        weights[i] = value;
        sum += value;
    }
    if (sum < 6.103515625e-5f) sum = 6.103515625e-5f;
    for (uint32_t i = 0; i < n_expert_used; i++)
        weights[i] = weights[i] / sum * expert_weight_scale;
    return 1;
}

int ds4_gpu_router_select_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    ds4_gpu_tensor *probs, const void *model_map, uint64_t model_size,
    uint64_t bias_offset, uint64_t hash_offset, uint32_t hash_rows,
    uint32_t token, uint32_t n_expert, uint32_t n_expert_used,
    float expert_weight_scale, uint32_t n_expert_groups, uint32_t n_group_used,
    bool has_bias, bool hash_mode, const ds4_gpu_tensor *logits)
{
    if (n_expert_groups > 1u || n_group_used > 0u) return 0;
    if (!selected || !weights || !probs || !logits || !model_map ||
        !selected->ptr || !weights->ptr || !probs->ptr || !logits->ptr) return 0;
    if (n_expert == 0 || n_expert_used == 0 || n_expert_used > n_expert) return 0;

    /* Decode supplies one router-logits row. The token ID is only relevant
     * to hash routing, which the engine applies immediately after this call. */
    if ((uint64_t)n_expert * sizeof(float) > logits->bytes) return 0;
    if ((uint64_t)n_expert_used * sizeof(int32_t) > selected->bytes) return 0;
    if ((uint64_t)n_expert_used * sizeof(float) > weights->bytes) return 0;
    if ((uint64_t)n_expert * sizeof(float) > probs->bytes) return 0;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (has_bias && !hash_mode) {
        if (model_size < sizeof(float)) return 0;
        if (bias_offset > model_size ||
            (uint64_t)n_expert * sizeof(float) > model_size - bias_offset) return 0;
        bias = (const float *)((const char *)model_map + bias_offset);
    }
    if (hash_mode) {
        const uint64_t hash_bytes = (uint64_t)hash_rows * n_expert_used * sizeof(int32_t);
        if (hash_offset > model_size || hash_bytes > model_size - hash_offset)
            return 0;
        hash = (const int32_t *)((const char *)model_map + hash_offset);
    }
    return ds4_router_select_row((int32_t *)selected->ptr, (float *)weights->ptr,
                                 (float *)probs->ptr, (const float *)logits->ptr,
                                 bias, hash, hash_rows, (int32_t)token,
                                 n_expert, n_expert_used, expert_weight_scale,
                                 hash_mode);
}

int ds4_gpu_router_select_batch_tensor(ds4_gpu_tensor *selected, ds4_gpu_tensor *weights,
    ds4_gpu_tensor *probs, const void *mm, uint64_t ms, uint64_t bo, uint64_t ho,
    uint32_t hr, uint32_t ng, uint32_t ngu, bool hb, bool hm,
    const ds4_gpu_tensor *logits, const ds4_gpu_tensor *tokens,
    uint32_t ne, uint32_t neu, float ws, uint32_t nt)
{
    if (ng > 1u || ngu > 0u) return 0;
    if (!selected || !weights || !probs || !logits || !tokens || !mm || nt == 0 ||
        ne == 0 || neu == 0 || neu > ne ||
        !selected->ptr || !weights->ptr || !probs->ptr ||
        !logits->ptr || !tokens->ptr) return 0;
    const uint64_t logits_count = (uint64_t)nt * ne;
    const uint64_t selected_count = (uint64_t)nt * neu;
    if (logits_count > UINT64_MAX / sizeof(float) ||
        selected_count > UINT64_MAX / sizeof(int32_t) ||
        (uint64_t)nt > UINT64_MAX / sizeof(int32_t)) return 0;
    if (logits->bytes < logits_count * sizeof(float) ||
        probs->bytes < logits_count * sizeof(float) ||
        selected->bytes < selected_count * sizeof(int32_t) ||
        weights->bytes < selected_count * sizeof(float) ||
        tokens->bytes < (uint64_t)nt * sizeof(int32_t)) return 0;
    const float *bias = NULL;
    const int32_t *hash = NULL;
    if (hb && !hm) {
        if (bo > ms || (uint64_t)ne * sizeof(float) > ms - bo) return 0;
        bias = (const float *)((const char *)mm + bo);
    }
    if (hm) {
        const uint64_t hash_count = (uint64_t)hr * neu;
        if (hash_count > UINT64_MAX / sizeof(int32_t)) return 0;
        const uint64_t hash_bytes = hash_count * sizeof(int32_t);
        if (ho > ms || hash_bytes > ms - ho) return 0;
        hash = (const int32_t *)((const char *)mm + ho);
    }
    const float *lp = (const float*)logits->ptr;
    const int32_t *token_ids = (const int32_t *)tokens->ptr;
    int32_t *sel = (int32_t*)selected->ptr;
    float *wp = (float*)weights->ptr;
    float *pp = (float *)probs->ptr;
    for (uint32_t t = 0; t < nt; t++) {
        if (!ds4_router_select_row(sel + (uint64_t)t * neu,
                                   wp + (uint64_t)t * neu,
                                   pp + (uint64_t)t * ne,
                                   lp + (uint64_t)t * ne,
                                   bias, hash, hr, token_ids[t], ne, neu, ws, hm))
            return 0;
    }
    return 1;
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
    VkDescriptorBufferInfo bufs[1] = {{xbuf, xoff, (VkDeviceSize)x->bytes}};
    struct { uint32_t n_rows; } pc = {n_rows};
    return record_simple_shader("indexer_qat", &pc, sizeof(pc), bufs, 1,
                                n_rows, 1, 1, resume_recording);
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
    if (!out || !out->ptr || !gate || !gate->ptr || !up || !up->ptr ||
        !mid || !mid->ptr || !experts || !experts->ptr ||
        !selected || !selected->ptr || !weights || !weights->ptr ||
        !x || !x->ptr || !model_map) {
        fprintf(stderr, "ds4: routed_moe_one: invalid argument\n");
        return 0;
    }
    if (n_expert == 0 || expert_in_dim == 0 || expert_mid_dim == 0 || out_dim == 0) {
        fprintf(stderr, "ds4: routed_moe_one: empty dimensions\n");
        return 0;
    }

    if (!ds4gk_routed_moe_slot(
            gate_type, down_type,
            expert_in_dim, expert_mid_dim, out_dim,
            n_total_expert, n_expert, clamp,
            (const uint8_t *)model_map, model_size,
            gate_offset, up_offset, down_offset,
            gate_expert_bytes, gate_row_bytes,
            down_expert_bytes, down_row_bytes,
            (const float *)x->ptr,
            (const int32_t *)selected->ptr,
            (const float *)weights->ptr,
            (float *)gate->ptr, (float *)up->ptr, (float *)mid->ptr,
            (float *)experts->ptr, (float *)out->ptr,
            "routed_moe_one")) {
        return 0;
    }

    if (add_in && add_in->ptr) {
        const float *ap = (const float *)add_in->ptr;
        float *op = (float *)out->ptr;
        for (uint32_t r = 0; r < out_dim; r++) op[r] += ap[r];
    }
    return 1;
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
    if (mid_is_f16) *mid_is_f16 = false;   /* host path writes f32 mid */
    if (!out || !out->ptr || !gate || !gate->ptr || !up || !up->ptr ||
        !mid || !mid->ptr || !experts || !experts->ptr ||
        !selected || !selected->ptr || !weights || !weights->ptr ||
        !x || !x->ptr || !model_map) {
        fprintf(stderr, "ds4: routed_moe_batch: invalid argument\n");
        return 0;
    }
    if (n_tokens == 0 || n_expert == 0 || expert_in_dim == 0 ||
        expert_mid_dim == 0 || out_dim == 0) {
        fprintf(stderr, "ds4: routed_moe_batch: empty dimensions\n");
        return 0;
    }
    if (gate_expert_bytes == 0 || gate_row_bytes == 0 ||
        down_expert_bytes == 0 || down_row_bytes == 0 ||
        (uint64_t)n_total_expert > UINT64_MAX / gate_expert_bytes ||
        (uint64_t)n_total_expert > UINT64_MAX / down_expert_bytes) {
        fprintf(stderr, "ds4: routed_moe_batch: expert byte-size overflow\n");
        return 0;
    }

    const uint8_t *base = (const uint8_t *)model_map;
    const uint64_t pair_stride = (uint64_t)n_expert * expert_mid_dim;
    const uint64_t exp_stride  = (uint64_t)n_expert * out_dim;

    for (uint32_t t = 0; t < n_tokens; t++) {
        if (!ds4gk_routed_moe_slot(
                gate_type, down_type,
                expert_in_dim, expert_mid_dim, out_dim,
                n_total_expert, n_expert, clamp,
                base, model_size,
                gate_offset, up_offset, down_offset,
                gate_expert_bytes, gate_row_bytes,
                down_expert_bytes, down_row_bytes,
                (const float *)x->ptr + (uint64_t)t * expert_in_dim,
                (const int32_t *)selected->ptr + (uint64_t)t * n_expert,
                (const float *)weights->ptr + (uint64_t)t * n_expert,
                (float *)gate->ptr     + (uint64_t)t * pair_stride,
                (float *)up->ptr       + (uint64_t)t * pair_stride,
                (float *)mid->ptr      + (uint64_t)t * pair_stride,
                (float *)experts->ptr  + (uint64_t)t * exp_stride,
                (float *)out->ptr      + (uint64_t)t * out_dim,
                "routed_moe_batch")) {
            return 0;
        }
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
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(gate, model_map, model_size,
                                             gate_offset, in_dim, out_dim, x, 1) != 0;
    if (ok) ok = ds4_gpu_matmul_q8_0_tensor(up, model_map, model_size,
                                             up_offset, in_dim, out_dim, x, 1) != 0;
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
        head_dim == 0 || ratio == 0 || ratio > 4u || n_tokens == 0 ||
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
    if (rows == 0 || width == 0 || width > 4096u || ratio == 0 || ratio > 4u ||
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
        head_dim == 0 || ratio == 0 || ratio > 4u || n_rot > head_dim ||
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
        head_dim == 0 || ratio == 0 || ratio > 4u || n_tokens == 0 ||
        n_rot > head_dim || (n_rot & 1u) != 0 ||
        (ape_type != 0u && ape_type != 1u) || norm_type != 0u || n_tokens > 65535u)
        return 0;
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
        (n_comp != 0 && comp_cache->bytes < comp_bytes)) return 0;
    set_model_map_identity(model_map, model_size);
    VkBuffer ape_buf; VkDeviceSize ape_off, ape_range;
    if (!find_model_buffer(ape_offset, ape_bytes, ape_buf, ape_off, ape_range)) return 0;
    if (!compressor_clear_vk(state_kv, state_score, state_rows * width,
                             0.0f, -INFINITY)) return 0;
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
    if (ok && n_comp != 0)
        ok = ds4_gpu_rms_norm_weight_rows_tensor(comp_cache, comp_cache, model_map,
                                                  model_size, norm_offset, head_dim,
                                                  n_comp, rms_eps);
    if (ok && n_comp != 0 && n_rot != 0)
        ok = compressor_rope_stride_vk(comp_cache, n_comp, head_dim, n_rot, pos0, ratio,
                                       n_ctx_orig, freq_base, freq_scale, ext_factor,
                                       attn_factor, beta_fast, beta_slow);
    if (ok && n_comp != 0 && quantize_fp8)
        ok = ds4_gpu_dsv4_fp8_kv_quantize_tensor(comp_cache, n_comp, head_dim, n_rot);
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
        in_dim == 0 || width == 0 || ratio == 0 || ratio > 4u ||
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
        auto it = g_vk.tensor_headers.find(t->ptr);
        if (it != g_vk.tensor_headers.end()) {
            vmaDestroyBuffer(g_vk.allocator, it->second->buffer, it->second->allocation);
            free(it->second);
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
            entry.last_gen = g_vk.cmd_gen;
            return true;
        }
    }
    return false;
}