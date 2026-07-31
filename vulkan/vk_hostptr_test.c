/* vk_hostptr_test.c — isolate crashes in vkGetMemoryHostPointerPropertiesEXT.
 *
 * Checks both loaders (vkGetInstanceProcAddr vs vkGetDeviceProcAddr) and both
 * handle types, on a small malloc and optionally on the model mmap.
 *
 * Build (from repo root):
 *   gcc -I<vulkan-include> vulkan/vk_hostptr_test.c -L<vulkan-lib> -lvulkan \
 *       -Wl,-rpath,<vulkan-lib> -o vk_hostptr_test
 * Run:
 *   VK_DRIVER_FILES=<icd>.json ./vk_hostptr_test [MODEL.gguf]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vulkan/vulkan.h>

typedef VkResult (VKAPI_PTR *PFN_HP)(VkDevice, VkExternalMemoryHandleTypeFlagBits,
                                     const void*, VkMemoryHostPointerPropertiesEXT*);

static void test(const char *label, const char *load, PFN_HP pfn, VkDevice dev,
                 const void *ptr) {
    const VkExternalMemoryHandleTypeFlagBits types[] = {
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
        VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_MAPPED_FOREIGN_MEMORY_BIT_EXT,
    };
    const char *names[] = { "HOST_ALLOCATION", "HOST_MAPPED_FOREIGN" };
    for (int i = 0; i < 2; i++) {
        VkMemoryHostPointerPropertiesEXT hpp = { VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
        VkResult r = pfn(dev, types[i], ptr, &hpp);
        printf("%s [%s/%s]: r=%d typeBits=0x%x\n", label, load, names[i],
               (int)r, (unsigned)hpp.memoryTypeBits);
        fflush(stdout);
    }
}

int main(int argc, char **argv) {
    VkInstance inst;
    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.apiVersion = VK_API_VERSION_1_3;
    const char *iexts[] = { VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME };
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &ai;
    ici.enabledExtensionCount = 1; ici.ppEnabledExtensionNames = iexts;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) { printf("instance fail\n"); return 1; }
    PFN_HP pfn_inst = (PFN_HP)vkGetInstanceProcAddr(inst, "vkGetMemoryHostPointerPropertiesEXT");
    printf("pfn(instance)=%p\n", (void*)pfn_inst);
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    if (n == 0) { printf("no devices\n"); return 1; }
    VkPhysicalDevice dev;
    vkEnumeratePhysicalDevices(inst, &n, &dev);
    VkPhysicalDeviceProperties p; vkGetPhysicalDeviceProperties(dev, &p);
    printf("device: %s\n", p.deviceName);
    uint32_t qc = 0; vkGetPhysicalDeviceQueueFamilyProperties(dev, &qc, NULL);
    VkQueueFamilyProperties *qps = calloc(qc, sizeof(*qps));
    vkGetPhysicalDeviceQueueFamilyProperties(dev, &qc, qps);
    int qf = -1;
    for (uint32_t i = 0; i < qc; i++)
        if (qps[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qf = i; break; }
    if (qf < 0) { printf("no compute queue\n"); return 1; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = { VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qci.queueFamilyIndex = qf; qci.queueCount = 1; qci.pQueuePriorities = &prio;
    const char *dexts[] = { VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME };
    VkDeviceCreateInfo dci = { VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dci.queueCreateInfoCount = 1; dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = 1; dci.ppEnabledExtensionNames = dexts;
    VkDevice dev2;
    if (vkCreateDevice(dev, &dci, NULL, &dev2) != VK_SUCCESS) { printf("device fail\n"); return 1; }
    PFN_HP pfn_dev = (PFN_HP)vkGetDeviceProcAddr(dev2, "vkGetMemoryHostPointerPropertiesEXT");
    printf("pfn(device)=%p\n", (void*)pfn_dev);

    void *small = malloc(1 << 20);
    printf("small malloc: %p\n", small);
    fflush(stdout);
    test("small malloc 1MiB", "instance", pfn_inst, dev2, small);
    test("small malloc 1MiB", "device", pfn_dev, dev2, small);

    if (argc > 1) {
        const char *path = argv[1];
        int fd = open(path, O_RDONLY);
        if (fd < 0) { perror("open"); return 1; }
        struct stat st; fstat(fd, &st);
        void *map = mmap(NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (map == MAP_FAILED) { perror("mmap"); return 1; }
        printf("model mmap %s: %p size=%llu\n", path, map, (unsigned long long)st.st_size);
        fflush(stdout);
        test("model mmap (whole 86G)", "instance", pfn_inst, dev2, map);
        test("model mmap (whole 86G)", "device", pfn_dev, dev2, map);
    }
    printf("done\n");
    return 0;
}
