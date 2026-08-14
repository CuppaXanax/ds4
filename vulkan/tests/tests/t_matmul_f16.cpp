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
#include <string>
#include <vector>

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
                            std::fabs(outv[t * out_dim + o] - ref[t * out_dim + o]) <= 1e-3f ? "ok" : "MISMATCH");
                }
            }
            rc = 0;
            for (uint64_t t = 0; t < n_tok && rc == 0; t++) {
                for (uint64_t o = 0; o < out_dim; o++) {
                    float got = outv[t * out_dim + o];
                    float want = ref[t * out_dim + o];
                    if (!(std::fabs(got - want) <= 1e-3f)) rc = 1;
                }
            }
        }
    }

    free(model);
    ds4_gpu_tensor_free(out);
    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(matmul_f16, test_matmul_f16);

static const char *k_native_unpack_env = "DS4_VULKAN_F16_NATIVE_UNPACK";

static int select_native_unpack(bool native) {
#if defined(_WIN32)
    return _putenv_s(k_native_unpack_env, native ? "1" : "");
#else
    return native ? setenv(k_native_unpack_env, "1", 1)
                  : unsetenv(k_native_unpack_env);
#endif
}

/* Tests force both paths but must not change the caller's measurement mode. */
class scoped_native_unpack_env {
public:
    scoped_native_unpack_env() {
        const char *value = std::getenv(k_native_unpack_env);
        present_ = value != nullptr;
        if (value) value_ = value;
    }
    ~scoped_native_unpack_env() {
#if defined(_WIN32)
        _putenv_s(k_native_unpack_env, present_ ? value_.c_str() : "");
#else
        if (present_) setenv(k_native_unpack_env, value_.c_str(), 1);
        else unsetenv(k_native_unpack_env);
#endif
    }

private:
    bool present_ = false;
    std::string value_;
};

static uint32_t f32_bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void store_le16(unsigned char *dst, uint16_t value) {
    dst[0] = (unsigned char)(value & 0xffu);
    dst[1] = (unsigned char)(value >> 8);
}

static bool run_matmul_f16_path(
        bool native,
        ds4_gpu_tensor *out,
        const std::vector<unsigned char> &model,
        uint64_t weight_offset,
        uint64_t in_dim,
        uint64_t out_dim,
        const ds4_gpu_tensor *x,
        uint64_t n_tok,
        std::vector<float> &values) {
    if (select_native_unpack(native) != 0) return false;
    if (ds4_gpu_matmul_f16_tensor(out, model.data(), model.size(),
                                  weight_offset, in_dim, out_dim, x,
                                  n_tok) == 0 ||
        ds4_gpu_synchronize() == 0) {
        return false;
    }
    values.resize(n_tok * out_dim);
    return ds4_gpu_tensor_read(out, 0, values.data(),
                               values.size() * sizeof(float)) != 0;
}

/* Exercise every IEEE-754 binary16 bit pattern through the real shader.  Two
 * 32,768-row, one-column matrices avoid the backend's 65,534-workgroup chunk
 * limit while still testing both low-word (.x/even) and high-word (.y/odd)
 * selection.  The matmul reduction canonicalizes -0 to +0.  NaN payload and
 * sign are explicitly excluded from bit-exact comparison because valid GGUF
 * model weights are finite and the legacy path canonicalizes every NaN; both
 * paths must still classify all 2,046 NaN encodings as NaN.  Every other
 * result must be bit-exact. */
static int test_matmul_f16_unpack_all_bits(void) {
    scoped_native_unpack_env restore_env;
    const uint64_t rows = 32768;
    const uint64_t bank_bytes = rows * sizeof(uint16_t);
    const uint64_t header = 4096;
    const uint64_t model_size = header + 2u * bank_bytes;
    std::vector<unsigned char> model(model_size, 0xA5);

    for (uint32_t bank = 0; bank < 2; bank++) {
        unsigned char *packed =
            model.data() + header + (uint64_t)bank * bank_bytes;
        for (uint32_t row = 0; row < rows; row++) {
            const uint16_t half_bits = (uint16_t)(bank * rows + row);
            store_le16(packed + 2u * row, half_bits);
        }
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(sizeof(float));
    ds4_gpu_tensor *native_out = ds4_gpu_tensor_alloc(rows * sizeof(float));
    ds4_gpu_tensor *manual_out = ds4_gpu_tensor_alloc(rows * sizeof(float));
    if (!x || !native_out || !manual_out) {
        if (manual_out) ds4_gpu_tensor_free(manual_out);
        if (native_out) ds4_gpu_tensor_free(native_out);
        if (x) ds4_gpu_tensor_free(x);
        return 1;
    }

    const float one = 1.0f;
    int rc = 0;
    if (ds4_gpu_set_model_map(model.data(), model.size()) == 0 ||
        ds4_gpu_tensor_write(x, 0, &one, sizeof(one)) == 0) {
        rc = 1;
    }

    std::vector<float> native_values, manual_values;
    for (uint32_t bank = 0; bank < 2 && rc == 0; bank++) {
        const uint64_t offset = header + (uint64_t)bank * bank_bytes;
        if (!run_matmul_f16_path(true, native_out, model, offset, 1, rows,
                                 x, 1, native_values) ||
            !run_matmul_f16_path(false, manual_out, model, offset, 1, rows,
                                 x, 1, manual_values)) {
            fprintf(stderr, "matmul_f16 all-bits: dispatch/read failed for bank %u\n",
                    bank);
            rc = 1;
            break;
        }

        for (uint32_t row = 0; row < rows; row++) {
            const uint16_t half_bits = (uint16_t)(bank * rows + row);
            const uint32_t exponent = (half_bits >> 10) & 0x1fu;
            const uint32_t mantissa = half_bits & 0x3ffu;
            const float native_value = native_values[row];
            const float manual_value = manual_values[row];

            if (exponent == 31u && mantissa != 0u) {
                if (!std::isnan(native_value) || !std::isnan(manual_value)) {
                    fprintf(stderr,
                            "matmul_f16 all-bits: 0x%04x NaN policy mismatch "
                            "native=0x%08x manual=0x%08x\n",
                            half_bits, f32_bits(native_value),
                            f32_bits(manual_value));
                    rc = 1;
                    break;
                }
                continue;
            }

            if ((half_bits & 0x7fffu) == 0u) {
                if (native_value != 0.0f || manual_value != 0.0f) {
                    fprintf(stderr,
                            "matmul_f16 all-bits: 0x%04x zero policy mismatch "
                            "native=0x%08x manual=0x%08x\n",
                            half_bits, f32_bits(native_value),
                            f32_bits(manual_value));
                    rc = 1;
                    break;
                }
                continue;
            }

            const uint32_t expected_bits = f32_bits(f16_to_f32(half_bits));
            if (f32_bits(native_value) != expected_bits ||
                f32_bits(manual_value) != expected_bits ||
                f32_bits(native_value) != f32_bits(manual_value)) {
                fprintf(stderr,
                        "matmul_f16 all-bits: 0x%04x mismatch "
                        "native=0x%08x manual=0x%08x expected=0x%08x\n",
                        half_bits, f32_bits(native_value),
                        f32_bits(manual_value), expected_bits);
                rc = 1;
                break;
            }
        }
    }

    ds4_gpu_tensor_free(manual_out);
    ds4_gpu_tensor_free(native_out);
    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(matmul_f16_unpack_all_bits, test_matmul_f16_unpack_all_bits);

/* Production router projection shape: 4096 activations -> 256 logits.  The
 * conversion changes, but every multiply, per-lane partial, float store, and
 * lane-0 reduction remains in the same order, so outputs must match exactly. */
static int test_matmul_f16_native_ab(void) {
    scoped_native_unpack_env restore_env;
    const uint64_t in_dim = 4096;
    const uint64_t out_dim = 256;
    const uint64_t weight_elems = in_dim * out_dim;
    const uint64_t header = 4096;
    const uint64_t weight_offset = header;
    std::vector<unsigned char> model(
        header + weight_elems * sizeof(uint16_t), 0x5A);
    for (uint64_t i = 0; i < weight_elems; i++) {
        int value = (int)((i * 37u + i / in_dim * 17u) % 2047u) - 1023;
        store_le16(model.data() + header + i * sizeof(uint16_t),
                   f32_to_f16((float)value / 1024.0f));
    }

    std::vector<float> input(in_dim);
    for (uint64_t i = 0; i < in_dim; i++) {
        int value = (int)((i * 29u) % 127u) - 63;
        input[i] = (float)value / 32.0f;
    }

    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc(in_dim * sizeof(float));
    ds4_gpu_tensor *native_out = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    ds4_gpu_tensor *manual_out = ds4_gpu_tensor_alloc(out_dim * sizeof(float));
    if (!x || !native_out || !manual_out) {
        if (manual_out) ds4_gpu_tensor_free(manual_out);
        if (native_out) ds4_gpu_tensor_free(native_out);
        if (x) ds4_gpu_tensor_free(x);
        return 1;
    }

    int rc = 0;
    if (ds4_gpu_set_model_map(model.data(), model.size()) == 0 ||
        ds4_gpu_tensor_write(x, 0, input.data(),
                             input.size() * sizeof(float)) == 0) {
        rc = 1;
    }

    std::vector<float> native_values, manual_values, warm_values;
    /* One warm dispatch per pipeline makes the following 16 interleaved pairs
     * same-binary and cache-hot.  With timeline skip=2/count=32, the focused
     * harness reports 16 native and 16 legacy production-shape GPU samples. */
    if (rc == 0 &&
        (!run_matmul_f16_path(true, native_out, model, weight_offset,
                              in_dim, out_dim, x, 1, warm_values) ||
         !run_matmul_f16_path(false, manual_out, model, weight_offset,
                              in_dim, out_dim, x, 1, warm_values))) {
        fprintf(stderr, "matmul_f16 native A/B: warm-up failed\n");
        rc = 1;
    }

    for (uint32_t pair = 0; pair < 16u && rc == 0; pair++) {
        if (!run_matmul_f16_path(true, native_out, model, weight_offset,
                                 in_dim, out_dim, x, 1, native_values) ||
            !run_matmul_f16_path(false, manual_out, model, weight_offset,
                                 in_dim, out_dim, x, 1, manual_values)) {
            fprintf(stderr,
                    "matmul_f16 native A/B: measured pair %u failed\n", pair);
            rc = 1;
            break;
        }
        if (std::memcmp(native_values.data(), manual_values.data(),
                        out_dim * sizeof(float)) == 0) {
            continue;
        }
        for (uint64_t i = 0; i < out_dim; i++) {
            if (f32_bits(native_values[i]) != f32_bits(manual_values[i])) {
                fprintf(stderr,
                        "matmul_f16 native A/B: pair %u row %llu "
                        "native=0x%08x manual=0x%08x\n",
                        pair, (unsigned long long)i,
                        f32_bits(native_values[i]),
                        f32_bits(manual_values[i]));
                break;
            }
        }
        rc = 1;
    }

    ds4_gpu_tensor_free(manual_out);
    ds4_gpu_tensor_free(native_out);
    ds4_gpu_tensor_free(x);
    return rc;
}
REGISTER_TEST(matmul_f16_native_ab, test_matmul_f16_native_ab);
