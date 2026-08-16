#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ds4.h"
#include "ds4_gpu.h"

#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { SLICE_START = 4, SLICE_END = 7, DEFAULT_CTX = 8192,
       DEFAULT_WARMUP = 1, DEFAULT_ITERS = 5 };

typedef struct {
    const char *model;
    const char *prompt_file;
    int ctx;
    uint32_t chunk;
    int warmup;
    int iters;
    int decode_token;
} config;

typedef struct {
    double prefill_wall_ms;
    double decode_wall_ms;
    double gpu_ms;
    double submit_cpu_ms;
    double wait_cpu_ms;
    uint64_t dispatches;
    uint64_t barriers;
    uint64_t submissions;
    uint64_t waits;
} metrics;

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static uint64_t hash_bytes(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = (const unsigned char *)p;
    for (size_t i = 0; i < n; i++) {
        h ^= b[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static uint64_t hash_f32(const float *p, uint64_t n) {
    return hash_bytes(UINT64_C(1469598103934665603), p,
                      (size_t)n * sizeof(*p));
}

static void usage(FILE *fp) {
    fprintf(fp,
            "usage: ds4-slice-bench --model FILE --prompt-file FILE [options]\n"
            "  --ctx N             context size (default %d)\n"
            "  --chunk N           production prefill chunk (default session cap)\n"
            "  --warmup N          untimed replays (default %d)\n"
            "  --iters N           measured replays (default %d)\n"
            "  --decode-token ID   token fed to the one-token decode\n"
            "\n"
            "Runs the real layer-slice API on one GPU. The measured route is\n"
            "layers %u:%u; source layers 0:3 produce the hidden input.\n",
            DEFAULT_CTX, DEFAULT_WARMUP, DEFAULT_ITERS, SLICE_START, SLICE_END);
}

static int parse_int(const char *s, const char *name, int min, int *out) {
    char *end = NULL;
    long v = strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < min || v > INT32_MAX) {
        fprintf(stderr, "ds4-slice-bench: invalid %s: %s\n", name, s);
        return 0;
    }
    *out = (int)v;
    return 1;
}

static int parse_u32(const char *s, const char *name, uint32_t *out) {
    char *end = NULL;
    unsigned long v = strtoul(s, &end, 10);
    if (!s[0] || !end || *end || v == 0 || v > UINT32_MAX) {
        fprintf(stderr, "ds4-slice-bench: invalid %s: %s\n", name, s);
        return 0;
    }
    *out = (uint32_t)v;
    return 1;
}

static int parse_args(int argc, char **argv, config *c) {
    *c = (config){ .ctx = DEFAULT_CTX, .warmup = DEFAULT_WARMUP,
                   .iters = DEFAULT_ITERS, .decode_token = -1 };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (!strcmp(arg, "--help") || !strcmp(arg, "-h")) {
            usage(stdout);
            return 0;
        }
        if (i + 1 >= argc) {
            fprintf(stderr, "ds4-slice-bench: %s requires an argument\n", arg);
            return -1;
        }
        const char *value = argv[++i];
        if (!strcmp(arg, "--model")) c->model = value;
        else if (!strcmp(arg, "--prompt-file")) c->prompt_file = value;
        else if (!strcmp(arg, "--ctx")) {
            if (!parse_int(value, "--ctx", 2, &c->ctx)) return -1;
        } else if (!strcmp(arg, "--chunk")) {
            if (!parse_u32(value, "--chunk", &c->chunk)) return -1;
        } else if (!strcmp(arg, "--warmup")) {
            if (!parse_int(value, "--warmup", 0, &c->warmup)) return -1;
        } else if (!strcmp(arg, "--iters")) {
            if (!parse_int(value, "--iters", 1, &c->iters)) return -1;
        } else if (!strcmp(arg, "--decode-token")) {
            if (!parse_int(value, "--decode-token", 0, &c->decode_token)) return -1;
        } else {
            fprintf(stderr, "ds4-slice-bench: unknown option %s\n", arg);
            return -1;
        }
    }
    if (!c->model || !c->prompt_file) {
        fprintf(stderr, "ds4-slice-bench: --model and --prompt-file are required\n");
        return -1;
    }
    return 1;
}

static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "ds4-slice-bench: open %s: %s\n", path, strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long n = ftell(fp);
    if (n < 0 || (uint64_t)n > SIZE_MAX - 1u) { fclose(fp); return NULL; }
    rewind(fp);
    char *buf = malloc((size_t)n + 1u);
    if (!buf || fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        free(buf); fclose(fp); return NULL;
    }
    buf[n] = '\0';
    fclose(fp);
    return buf;
}

static int fail(const char *where, const char *err) {
    fprintf(stderr, "ds4-slice-bench: %s: %s\n", where,
            err && err[0] ? err : "operation failed");
    return 0;
}

/* Capture the coordinator-produced hidden input once. This is deliberately
 * outside the measured loop: the benchmark must report the 4:7 worker slice,
 * not the coordinator's 0:3 work or its CPU handoff. */
static int capture_source_inputs(ds4_session *source, const ds4_tokens *prompt,
                                 uint32_t chunk, uint64_t hidden,
                                 float *inputs) {
    char err[512] = {0};
    unsetenv("DS4_VULKAN_TIMELINE_LAYER");
    for (uint32_t pos = 0; pos < (uint32_t)prompt->len; ) {
        uint32_t n = (uint32_t)prompt->len - pos;
        if (n > chunk) n = chunk;
        if (ds4_session_eval_layer_slice(source, prompt->v + pos, n, pos,
                                          0, 3, NULL,
                                          inputs + (uint64_t)pos * hidden,
                                          false, NULL,
                                          err, sizeof(err)) != 0)
            return fail("capture source 0:3", err);
        pos += n;
    }
    return 1;
}

static int replay_target_prefill(ds4_session *target, const ds4_tokens *prompt,
                                 uint32_t chunk, uint64_t hidden,
                                 const float *inputs, float *target_hc,
                                 uint64_t *out_hash, metrics *m) {
    char err[512] = {0};
    const double t0 = now_ms();
    unsetenv("DS4_VULKAN_TIMELINE_LAYER");
    for (uint32_t pos = 0; pos < (uint32_t)prompt->len; ) {
        uint32_t n = (uint32_t)prompt->len - pos;
        if (n > chunk) n = chunk;
        const bool last = pos + n == (uint32_t)prompt->len;
        if (ds4_session_eval_layer_slice(
                target, prompt->v + pos, n, pos, SLICE_START, SLICE_END,
                inputs + (uint64_t)pos * hidden, last ? target_hc : NULL,
                false, NULL, err, sizeof(err)) != 0)
            return fail("replay target prefill 4:7", err);
        pos += n;
    }
    if (out_hash) *out_hash = hash_f32(target_hc, hidden);
    if (m) m->prefill_wall_ms += now_ms() - t0;
    return 1;
}

static int replay_target_decode(ds4_session *target, int token, uint32_t pos,
                                uint64_t hidden, const float *input_hc,
                                float *target_hc, uint64_t *out_hash,
                                metrics *m) {
    char err[512] = {0};
    const double t0 = now_ms();
    setenv("DS4_VULKAN_TIMELINE_LAYER", "4", 1);
    if (ds4_gpu_worker_slice_begin(SLICE_START, SLICE_END) == 0)
        return fail("decode worker slice begin", "Vulkan worker slice unavailable");
    const int rc = ds4_session_eval_layer_slice(target, &token, 1, pos,
                                                SLICE_START, SLICE_END,
                                                input_hc, target_hc, false,
                                                NULL, err, sizeof(err));
    const int end_rc = ds4_gpu_worker_slice_end(SLICE_START, SLICE_END);
    if (rc != 0) return fail("decode target 4:7", err);
    if (end_rc == 0) return fail("decode worker slice end", "retirement failed");
    if (out_hash) *out_hash = hash_f32(target_hc, hidden);
    if (m) {
        ds4_gpu_timeline_stats s = {0};
        ds4_gpu_timeline_layer_read_stats(&s);
        m->decode_wall_ms += now_ms() - t0;
        m->gpu_ms += s.gpu_ms;
        m->submit_cpu_ms += s.submit_cpu_ms;
        m->wait_cpu_ms += s.wait_cpu_ms;
        m->dispatches += s.dispatches;
        m->barriers += s.barriers;
        m->submissions += s.submissions;
        m->waits += s.waits;
    }
    return 1;
}

static int reset_target(ds4_session *target) {
    char err[256] = {0};
    if (ds4_session_layer_slice_reset(target, err, sizeof(err)) != 0)
        return fail("reset target", err);
    return 1;
}

int main(int argc, char **argv) {
    config c;
    const int parsed = parse_args(argc, argv, &c);
    if (parsed <= 0) return parsed == 0 ? 0 : 2;

    /* The worker-slice fast path is deliberately opt-in in production. This
     * command is that explicit opt-in, while retaining the production graph
     * and layer-slice payload shape. Timeline capture is enabled only for the
     * one-token worker decode so prefill remains unperturbed. */
    setenv("DS4_VULKAN_WORKER_SLICE_BATCH", "1", 1);
    setenv("DS4_VULKAN_SLICE_BENCH", "1", 1);
    setenv("DS4_VULKAN_TIMELINE_COUNT", "1024", 0);
    /* Query pools are created at engine startup for decode. replay_target_prefill()
     * removes this selector while running production-shaped prefill. */
    setenv("DS4_VULKAN_TIMELINE_LAYER", "4", 1);

    ds4_engine_options opt;
    memset(&opt, 0, sizeof(opt));
    opt.model_path = c.model;
    opt.backend = DS4_BACKEND_CUDA;
    opt.context_size = c.ctx;
    opt.prefill_chunk = c.chunk;
    opt.warm_weights = true;
    opt.load_slice = true;
    opt.load_layer_start = 0;
    opt.load_layer_end = SLICE_END;
    opt.load_output = false;
    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &opt) != 0 || !engine) return 2;
    if (ds4_engine_layer_count(engine) <= SLICE_END) {
        fprintf(stderr, "ds4-slice-bench: model has fewer than 8 executable layers\n");
        ds4_engine_close(engine);
        return 2;
    }

    char *prompt_text = read_file(c.prompt_file);
    if (!prompt_text) { ds4_engine_close(engine); return 2; }
    ds4_tokens prompt = {0};
    ds4_tokenize_rendered_chat(engine, prompt_text, &prompt);
    free(prompt_text);
    if (prompt.len <= 0 || prompt.len >= c.ctx) {
        fprintf(stderr, "ds4-slice-bench: prompt token count %d is invalid for ctx %d\n",
                prompt.len, c.ctx);
        ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
    }
    if (c.decode_token < 0) c.decode_token = prompt.v[prompt.len - 1];
    if (c.decode_token >= ds4_engine_vocab_size(engine)) {
        fprintf(stderr, "ds4-slice-bench: decode token %d is outside vocab\n", c.decode_token);
        ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
    }

    ds4_session *source = NULL, *target = NULL;
    if (ds4_session_create(&source, engine, c.ctx) != 0 ||
        ds4_session_create(&target, engine, c.ctx) != 0) {
        fprintf(stderr, "ds4-slice-bench: failed to create source/target sessions\n");
        ds4_session_free(source); ds4_session_free(target);
        ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
    }
    uint64_t hidden = ds4_engine_hidden_f32_values(engine);
    uint32_t chunk = c.chunk ? c.chunk : (uint32_t)ds4_session_prefill_cap(source);
    if (chunk == 0 || hidden == 0) {
        fprintf(stderr, "ds4-slice-bench: invalid session prefill cap or hidden size\n");
        ds4_session_free(source); ds4_session_free(target);
        ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
    }
    if (chunk > (uint32_t)prompt.len) chunk = (uint32_t)prompt.len;
    const uint64_t input_rows = (uint64_t)prompt.len + 1u;
    if (input_rows > SIZE_MAX / hidden ||
        input_rows * hidden > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "ds4-slice-bench: captured hidden-state input is too large\n");
        ds4_session_free(source); ds4_session_free(target);
        ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
    }
    float *captured_inputs = calloc((size_t)(input_rows * hidden), sizeof(*captured_inputs));
    float *target_hc = calloc((size_t)hidden, sizeof(*target_hc));
    if (!captured_inputs || !target_hc) {
        fprintf(stderr, "ds4-slice-bench: hidden-state allocation failed\n");
        free(captured_inputs); free(target_hc);
        ds4_session_free(source); ds4_session_free(target);
        ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
    }

    if (!capture_source_inputs(source, &prompt, chunk, hidden, captured_inputs)) {
        free(captured_inputs); free(target_hc); ds4_session_free(source); ds4_session_free(target);
        ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
    }
    /* Capture the exact coordinator hidden input for the one-token decode. */
    {
        char err[512] = {0};
        const uint32_t pos = (uint32_t)prompt.len;
        int token = c.decode_token;
        if (ds4_session_eval_layer_slice(
                source, &token, 1, pos, 0, 3, NULL,
                captured_inputs + (uint64_t)pos * hidden, false, NULL,
                err, sizeof(err)) != 0) {
            fail("capture source decode 0:3", err);
            free(captured_inputs); free(target_hc);
            ds4_session_free(source); ds4_session_free(target);
            ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
        }
    }

    uint64_t capture_prefill_hash = 0, capture_decode_hash = 0;
    if (!replay_target_prefill(target, &prompt, chunk, hidden, captured_inputs,
                               target_hc, &capture_prefill_hash, NULL) ||
        !replay_target_decode(target, c.decode_token, (uint32_t)prompt.len,
                              hidden, captured_inputs + (uint64_t)prompt.len * hidden,
                              target_hc, &capture_decode_hash, NULL)) {
        free(captured_inputs); free(target_hc);
        ds4_session_free(source); ds4_session_free(target);
        ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
    }
    if (!reset_target(target)) {
        free(captured_inputs); free(target_hc);
        ds4_session_free(source); ds4_session_free(target);
        ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
    }

    for (int i = 0; i < c.warmup; i++) {
        if (!replay_target_prefill(target, &prompt, chunk, hidden, captured_inputs,
                                   target_hc, NULL, NULL) ||
            !replay_target_decode(target, c.decode_token, (uint32_t)prompt.len,
                                  hidden, captured_inputs + (uint64_t)prompt.len * hidden,
                                  target_hc, NULL, NULL) ||
            !reset_target(target)) {
            free(captured_inputs); free(target_hc);
            ds4_session_free(source); ds4_session_free(target);
            ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
        }
    }

    metrics m = {0};
    uint64_t measured_prefill_hash = 0, measured_decode_hash = 0;
    for (int i = 0; i < c.iters; i++) {
        if (!replay_target_prefill(target, &prompt, chunk, hidden, captured_inputs,
                                   target_hc, &measured_prefill_hash, &m) ||
            !replay_target_decode(target, c.decode_token, (uint32_t)prompt.len,
                                  hidden, captured_inputs + (uint64_t)prompt.len * hidden,
                                  target_hc, &measured_decode_hash, &m) ||
            !reset_target(target)) {
            free(captured_inputs); free(target_hc);
            ds4_session_free(source); ds4_session_free(target);
            ds4_tokens_free(&prompt); ds4_engine_close(engine); return 2;
        }
        if (measured_prefill_hash != capture_prefill_hash ||
            measured_decode_hash != capture_decode_hash) {
            fprintf(stderr,
                    "ds4-slice-bench: output hash changed at iteration %d "
                    "(prefill=%016" PRIx64 "/%016" PRIx64 " decode=%016" PRIx64 "/%016" PRIx64 ")\n",
                    i, measured_prefill_hash, capture_prefill_hash,
                    measured_decode_hash, capture_decode_hash);
            free(captured_inputs); free(target_hc);
            ds4_session_free(source); ds4_session_free(target);
            ds4_tokens_free(&prompt); ds4_engine_close(engine); return 3;
        }
    }

    printf("slice_bench layers=%u:%u prompt_tokens=%d chunk=%u chunks=%u warmup=%d iters=%d decode_token=%d\n",
           SLICE_START, SLICE_END, prompt.len, chunk,
           ((uint32_t)prompt.len + chunk - 1u) / chunk,
           c.warmup, c.iters, c.decode_token);
    printf("capture_hash prefill=%016" PRIx64 " decode=%016" PRIx64 "\n",
           capture_prefill_hash, capture_decode_hash);
    printf("scope target_only=layers_%u:%u source_capture=untimed_0:3 "
           "input_hc_upload=each_chunk_and_decode output_hc_readback=final_prefill_and_decode\n",
           SLICE_START, SLICE_END);
    printf("timing prefill_wall_ms=%.3f decode_wall_ms=%.3f total_wall_ms=%.3f\n",
           m.prefill_wall_ms / c.iters, m.decode_wall_ms / c.iters,
           (m.prefill_wall_ms + m.decode_wall_ms) / c.iters);
    printf("decode_gpu_ms=%.3f submit_cpu_ms=%.3f wait_cpu_ms=%.3f "
           "dispatches=%.1f barriers=%.1f submissions=%.1f waits=%.1f\n",
           m.gpu_ms / c.iters, m.submit_cpu_ms / c.iters, m.wait_cpu_ms / c.iters,
           (double)m.dispatches / c.iters, (double)m.barriers / c.iters,
           (double)m.submissions / c.iters, (double)m.waits / c.iters);
    printf("hash_check=PASS\n");

    free(captured_inputs); free(target_hc);
    ds4_session_free(source); ds4_session_free(target);
    ds4_tokens_free(&prompt); ds4_engine_close(engine);
    return 0;
}
