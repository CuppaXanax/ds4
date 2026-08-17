# Benchmark B — production-layout decode roofline

This probe targets `DS4_SHAPE_FLASH`: 4096 input, 2048 expert-mid, 4096
output, 256 experts, top-6. Gate and up are IQ2_XXS; down is Q2_K. A separate
Q8_0 stream (1024→32768, 34 bytes per 32-value row block) is decoded in the
same benchmark sample with the aligned-bfe prequant activation ABI (32×36
bytes), so Q8, IQ2, and Q2 traffic is reported independently.
Each sample has one Q8_0 dispatch, one IQ2 gate/up dispatch, and one Q2 down
dispatch; it performs the production unpack/dequant loops and reduces only to
checksum buffers (there is no output tensor and no layer orchestration).

Build the shader and host on the target Vulkan machine:

```sh
GLSLANG=glslangValidator python3 vulkan/shaders/compile.py
g++ -O3 -std=c++17 -Ivulkan/include speed-bench/vulkan_roofline_b_layout_decode.cpp \
    -lvulkan -o speed-bench/vulkan_roofline_b_layout_decode
speed-bench/vulkan_roofline_b_layout_decode 10
```

The output reports separate useful bytes for Q8 input, IQ2 gate, IQ2 up, and
Q2 down, the aggregate top-6 weight bytes, GPU timestamp duration, checksum,
and the warm-upload proof. Clock and temperature are reported when the Linux
driver exposes them; otherwise they remain `unknown`. Do not run this probe
concurrently with another Vulkan workload during the serialized measurement.
