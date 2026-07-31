/* vk_mem_info.c — tiny Vulkan memory-properties dumper.
 *
 * Diagnoses the GPU heap layout (useful for iGPU / unified-memory setups:
 * how much memory RADV actually exposes, and whether host-visible memory is
 * a separate, larger heap).
 *
 * Build:
 *   gcc -I<vulkan-include> vulkan/vk_mem_info.c -L<vulkan-lib> -lvulkan \
 *       -Wl,-rpath,<vulkan-lib> -o /tmp/vk_mem_info
 * Run (adjust VK_DRIVER_FILES to a valid ICD manifest if the loader cannot
 * discover one automatically):
 *   VK_DRIVER_FILES=<icd>.json ./vk_mem_info
 */
#include <stdio.h>
#include <vulkan/vulkan.h>

int main(void) {
    VkInstance inst = VK_NULL_HANDLE;
    VkApplicationInfo ai = { VK_STRUCTURE_TYPE_APPLICATION_INFO };
    ai.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ici = { VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    ici.pApplicationInfo = &ai;
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) {
        fprintf(stderr, "instance failed\n");
        return 1;
    }
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, NULL);
    VkPhysicalDevice devs[8];
    n = n < 8 ? n : 8;
    vkEnumeratePhysicalDevices(inst, &n, devs);
    for (uint32_t d = 0; d < n; d++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(devs[d], &p);
        printf("=== %s (type %d) ===\n", p.deviceName, (int)p.deviceType);
        VkPhysicalDeviceMemoryProperties m;
        vkGetPhysicalDeviceMemoryProperties(devs[d], &m);
        printf("heaps: %u\n", m.memoryHeapCount);
        for (uint32_t h = 0; h < m.memoryHeapCount; h++) {
            printf("  heap %u: size=%.2f GiB flags=0x%x (%s%s)\n", h,
                   (double)m.memoryHeaps[h].size / (1024.0*1024.0*1024.0),
                   (unsigned)m.memoryHeaps[h].flags,
                   (m.memoryHeaps[h].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "",
                   (m.memoryHeaps[h].flags & VK_MEMORY_HEAP_MULTI_INSTANCE_BIT) ? "MULTI_INSTANCE" : "");
        }
        printf("memory types: %u\n", m.memoryTypeCount);
        for (uint32_t t = 0; t < m.memoryTypeCount; t++) {
            printf("  type %u: heap=%u flags=0x%x (%s%s%s%s)\n", t,
                   m.memoryTypes[t].heapIndex, (unsigned)m.memoryTypes[t].propertyFlags,
                   (m.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "",
                   (m.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "HOST_VISIBLE " : "",
                   (m.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "HOST_COHERENT " : "",
                   (m.memoryTypes[t].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "HOST_CACHED" : "");
        }
    }
    vkDestroyInstance(inst, NULL);
    return 0;
}
