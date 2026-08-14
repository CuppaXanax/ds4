/* Kernel test: ds4_gpu_matmul_f16_tensor (real GPU dispatch of matmul_f16).
 *
 * out[t][o] = sum_i x[t][i] * W_f16[o][i]
 *
 * where W is an IEEE half-precision (f16) matrix stored in the "model"
 * buffer as 2 bytes per element.  The model buffer mimics a GGUF-style
 * file: a small junk header, then the raw f16 weights at weight_offset.
 * The CPU reference decodes the stored f16 back to f32 exactly like the
 * shader does, so this validates the full storage + decode + matmul path.
 */
#include "../tests.h"
#include "../../ds4_gpu.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cstdio>
#include <cstdlib>

static void set_layer_batch_span(const char *value) {
#if defined(_WIN32)
    _putenv_s("DS4_VULKAN_BATCH_LAYER_SPAN", value);
#else
    setenv("DS4_VULKAN_BATCH_LAYER_SPAN", value, 1);
#endif
}

/* f32 -> IEEE half (round toward zero; test values are normal-range). */
static uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    uint32_t e    = (x >> 23) & 0xffu;
    uint32_t m    = x & 0x7fffffu;
    if (e == 0xffu) return (uint16_t)(sign | 0x7c00u | (m ? 0x200u : 0u));
    if (e == 0u)    return (uint16_t)sign;              /* ±0 / flush subnormal */
    int32_t e16 = (int32_t)e - 127 + 15;
    if (e16 >= 31)  return (uint16_t)(sign | 0x7c00u);  /* ±inf */
    if (e16 <= 0) {                                     /* subnormal f16 */
        uint32_t shift = 126u - e;
        uint32_t m16 = (0x800000u | m) >> shift;
        return (uint16_t)(sign | m16);
    }
    return (uint16_t)(sign | ((uint32_t)e16 << 10) | (m >> 13));
}

/* IEEE half -> f32 (exact decode, same semantics as the shader). */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e    = (h >> 10) & 0x1fu;
    uint32_t m    = h & 0x3ffu;
    uint32_t bits;
    if (e == 0u) {
        if (m == 0u) bits = sign;                       /* ±0 */
        else {                                          /* subnormal */
            uint32_t mm = m, p = 0u;
            while (!(mm & 0x400u) && p < 10u) { mm <<= 1; p++; }
            bits = sign | ((113u - p) << 23) | ((mm & 0x3ffu) << 13);
        }
    } else if (e == 31u) {
        bits = sign | 0x7f800000u | (m << 13);          /* ±inf/nan */
    } else {
        bits = sign | ((e + 112u) << 23) | (m << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

static int test_matmul_f16(void) {
    const uint64_t in_dim   = 8;
    const uint64_t out_dim  = 4;
    const uint64_t n_tok    = 2;
    const uint64_t w_elems  = out_dim * in_dim;         /* 32 f16 values */
    const uint64_t w_bytes  = w_elems * 2;              /* 2 bytes per f16 */
    const uint64_t header   = 16;                       /* mimic file header */
    const uint64_t model_size = header + w_bytes;
    const uint64_t weight_offset = header;

    ds4_gpu_tensor *x   = ds4_gpu_tensor_alloc(n_tok * in_dim * sizeof(float));
    ds4_gpu_tensor *out = ds4_gpu_tensor_alloc(n_tok * out_dim * sizeof(float));
    if (!x || !out) {
        if (x) ds4_gpu_tensor_free(x);
        if (out) ds4_gpu_tensor_free(out);
        return 1;
    }

    /* Synthetic "model file": junk header + f16 weight matrix. */
    unsigned char *model = (unsigned char*)malloc(model_size);
    if (!model) { ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out); return 1; }
    std::memset(model, 0xAA, header);
    uint16_t *w16 = (uint16_t*)(model + header);
    for (uint64_t o = 0; o < out_dim; o++) {
        for (uint64_t i = 0; i < in_dim; i++) {
            float v = (float)((o + 1) * (i + 1)) * 0.25f;
            if (((o + i) & 1u) != 0) v = -v;            /* alternate signs */
            w16[o * in_dim + i] = f32_to_f16(v);
        }
    }
    if (ds4_gpu_set_model_map(model, model_size) == 0) {
        free(model); ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out); return 1;
    }

    /* Known input activations (f32). */
    float xv[n_tok * in_dim];
    for (uint64_t t = 0; t < n_tok; t++)
        for (uint64_t i = 0; i < in_dim; i++)
            xv[t * in_dim + i] = (float)(t * 8 + i + 1) * 0.125f;
    if (ds4_gpu_tensor_write(x, 0, xv, sizeof(xv)) == 0) {
        free(model); ds4_gpu_tensor_free(x); ds4_gpu_tensor_free(out); return 1;
    }

    /* CPU reference: decode stored f16, accumulate in double like the shader. */
    float ref[n_tok * out_dim];
    for (uint64_t t = 0; t < n_tok; t++) {
        for (uint64_t o = 0; o < out_dim; o++) {
            double acc = 0.0;
            for (uint64_t i = 0; i < in_dim; i++)
                acc += (double)f16_to_f32(w16[o * in_dim + i]) *
                       (double)xv[t * in_dim + i];
            ref[t * out_dim + o] = (float)acc;
        }
    }

    int rc = 1;
    if (ds4_gpu_begin_commands() != 0 &&
        ds4_gpu_matmul_f16_tensor(out, model, model_size, weight_offset,
                                  in_dim, out_dim, x, n_tok) != 0 &&
        ds4_gpu_end_commands() != 0) {
        ds4_gpu_synchronize();              /* wait for GPU before reading */
        float outv[n_tok * out_dim];
        if (ds4_gpu_tensor_read(out, 0, outv, sizeof(outv)) != 0) {
            fprintf(stderr, "--- matmul_f16 diagnostic ---\n");
            for (uint64_t o = 0; o < out_dim; o++) {
                fprintf(stderr, "W[%llu]: ", (unsigned long long)o);
                for (uint64_t i = 0; i < in_dim; i++)
                    fprintf(stderr, "%.3f ", f16_to_f32(w16[o * in_dim + i]));
                fprintf(stderr, "\n");
            }
            fprintf(stderr, "X[0]: ");
            for (uint64_t i = 0; i < in_dim; i++)
                fprintf(stderr, "%.3f ", xv[i]);
            fprintf(stderr, "\n");
            for (uint64_t t = 0; t < n_tok; t++) {
                for (uint64_t o = 0; o < out_dim; o++) {
                    fprintf(stderr, "out[%llu][%llu] got=%.6f want=%.6f %s\n",
                            (unsigned long long)t, (unsigned long long)o,
                            outv[t * out_dim + o], ref[t * out_dim + o],
                            std::fabsf(outv[t * out_dim + o] - ref[t * out_dim + o]) <= 1e-3f ? "ok" : "MISMATCH");
                }
            }
            rc = 0;
            for (uint64_t t = 0; t < n_tok && rc == 0; t++) {
                for (uint64_t o = 0; o < out_dim; o++) {
                    float got = outv[t * out_dim + o];
                    float want = ref[t * out_dim + o];
                    if (!(std::fabsf(got - want) <= 1e-3f)) rc = 1;
                }
            }
        }
    }

    /* Cross the backend's 64-dispatch command-buffer rotation while a layer
     * span is open.  The rotation must fence and retire prior descriptor/temp
     * generations, then restore the exact layer/HC guard before dispatch 65. */
    if (rc == 0) {
        set_layer_batch_span("4");
        int rotation_ok = ds4_gpu_begin_commands() != 0 &&
            ds4_gpu_batch_layer_begin(17, x, out) != 0;
        for (uint32_t dispatch = 0; rotation_ok && dispatch < 65; dispatch++) {
            rotation_ok = ds4_gpu_matmul_f16_tensor(
                out, model, model_size, weight_offset,
                in_dim, out_dim, x, n_tok) != 0;
        }
        if (rotation_ok) {
            rotation_ok = ds4_gpu_batch_layer_end(17, x, out) != 0 &&
                ds4_gpu_end_commands() != 0;
        }
        if (!rotation_ok) (void)ds4_gpu_synchronize();
        float rotated[n_tok * out_dim];
        if (!rotation_ok ||
            ds4_gpu_tensor_read(out, 0, rotated, sizeof(rotated)) == 0) {
            rc = 1;
        } else {
            for (uint64_t i = 0; i < n_tok * out_dim; i++) {
                if (!(std::fabsf(rotated[i] - ref[i]) <= 1e-3f)) {
                    fprintf(stderr,
                            "matmul_f16 rotation mismatch at %llu got=%g want=%g\n",
                            (unsigned long long)i,
                            (double)rotated[i], (double)ref[i]);
                    rc = 1;
                    break;
                }
            }
        }
        set_layer_batch_span("1");
    }

    free(model);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(matmul_f16, test_matmul_f16);
