#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "ds4.h"
#include "ds4_gpu.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

enum { SOURCE_END = 3, SLICE_START = 4, SLICE_END = 7,
       DEFAULT_CTX = 8192, DEFAULT_WARMUP = 1, DEFAULT_ITERS = 5 };
#define CORPUS_MAGIC UINT32_C(0x43533444) /* D4SC, little endian */
#define CORPUS_VERSION UINT32_C(1)

#pragma pack(push, 1)
typedef struct {
    uint32_t magic, version, header_bytes, model_id;
    uint64_t model_name_hash, model_bytes, hidden_f32;
    uint32_t ctx_size, chunk_size, prompt_tokens, chunk_count;
    int32_t decode_token;
    uint32_t reserved;
    uint64_t chunk_offset, token_offset, prefill_hc_offset, decode_hc_offset;
    uint64_t file_bytes, payload_hash;
} corpus_header;
typedef struct { uint32_t pos0, n_tokens; uint64_t hc_row_offset; } corpus_chunk;
#pragma pack(pop)
_Static_assert(sizeof(corpus_header) == 112, "corpus header ABI");
_Static_assert(sizeof(corpus_chunk) == 16, "corpus chunk ABI");

typedef struct {
    const char *model, *prompt_file, *capture_file, *replay_file;
    int ctx, warmup, iters, decode_token;
    uint32_t chunk;
    bool ctx_set, chunk_set, decode_token_set;
} config;
typedef struct {
    double prefill_wall_ms, decode_wall_ms, gpu_ms, submit_cpu_ms, wait_cpu_ms;
    uint64_t dispatches, barriers, submissions, waits;
} metrics;
typedef struct {
    corpus_header h;
    int32_t *tokens;
    corpus_chunk *chunks;
    float *prefill_hc, *decode_hc;
} corpus;

static double now_ms(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}
static uint64_t hash_bytes(uint64_t h, const void *p, size_t n) {
    const unsigned char *b = p;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= UINT64_C(1099511628211); }
    return h;
}
static uint64_t hash_text(const char *s) {
    return hash_bytes(UINT64_C(1469598103934665603), s, strlen(s));
}
static uint64_t hash_f32(const float *p, uint64_t n) {
    if (n > SIZE_MAX / sizeof(*p)) return 0;
    return hash_bytes(UINT64_C(1469598103934665603), p, (size_t)n * sizeof(*p));
}
static int fail(const char *where, const char *why) {
    fprintf(stderr, "ds4-slice-bench: %s: %s\n", where,
            why && why[0] ? why : "operation failed");
    return 0;
}
static void usage(FILE *fp) {
    fprintf(fp,
        "usage:\n"
        "  ds4-slice-bench --model FILE --prompt-file FILE --capture CORPUS [options]\n"
        "  ds4-slice-bench --model FILE --replay CORPUS [options]\n\n"
        "capture loads only layers 0:%u and writes a versioned atomic corpus containing\n"
        "prompt tokens, chunk metadata, real prefill HC inputs, and one decode HC input.\n"
        "replay loads only layers %u:%u, validates model/shape/header, and times the target.\n\n"
        "  --ctx N             context (capture default %d; replay corpus value)\n"
        "  --chunk N           prefill chunk (capture default session cap)\n"
        "  --warmup N          untimed replays (default %d)\n"
        "  --iters N           measured replays (default %d)\n"
        "  --decode-token ID   capture decode token (default last prompt token)\n",
        SOURCE_END, SLICE_START, SLICE_END, DEFAULT_CTX, DEFAULT_WARMUP, DEFAULT_ITERS);
}
static int parse_int(const char *s, const char *name, int min, int *out) {
    char *end = NULL; long v = strtol(s, &end, 10);
    if (!s[0] || !end || *end || v < min || v > INT32_MAX) {
        fprintf(stderr, "ds4-slice-bench: invalid %s: %s\n", name, s); return 0;
    }
    *out = (int)v; return 1;
}
static int parse_u32(const char *s, const char *name, uint32_t *out) {
    char *end = NULL; unsigned long v = strtoul(s, &end, 10);
    if (!s[0] || !end || *end || v == 0 || v > UINT32_MAX) {
        fprintf(stderr, "ds4-slice-bench: invalid %s: %s\n", name, s); return 0;
    }
    *out = (uint32_t)v; return 1;
}
static int parse_args(int argc, char **argv, config *c) {
    *c = (config){ .ctx = DEFAULT_CTX, .warmup = DEFAULT_WARMUP,
                   .iters = DEFAULT_ITERS, .decode_token = -1 };
    for (int i = 1; i < argc; i++) {
        const char *a = argv[i];
        if (!strcmp(a, "--help") || !strcmp(a, "-h")) { usage(stdout); return 0; }
        if (i + 1 >= argc) { fprintf(stderr, "ds4-slice-bench: %s requires an argument\n", a); return -1; }
        const char *v = argv[++i];
        if (!strcmp(a, "--model")) c->model = v;
        else if (!strcmp(a, "--prompt-file")) c->prompt_file = v;
        else if (!strcmp(a, "--capture")) c->capture_file = v;
        else if (!strcmp(a, "--replay")) c->replay_file = v;
        else if (!strcmp(a, "--ctx")) { if (!parse_int(v, a, 2, &c->ctx)) return -1; c->ctx_set = true; }
        else if (!strcmp(a, "--chunk")) { if (!parse_u32(v, a, &c->chunk)) return -1; c->chunk_set = true; }
        else if (!strcmp(a, "--warmup")) { if (!parse_int(v, a, 0, &c->warmup)) return -1; }
        else if (!strcmp(a, "--iters")) { if (!parse_int(v, a, 1, &c->iters)) return -1; }
        else if (!strcmp(a, "--decode-token")) { if (!parse_int(v, a, 0, &c->decode_token)) return -1; c->decode_token_set = true; }
        else { fprintf(stderr, "ds4-slice-bench: unknown option %s\n", a); return -1; }
    }
    if (!c->model) { fprintf(stderr, "ds4-slice-bench: --model is required\n"); return -1; }
    if (!!c->capture_file == !!c->replay_file) { fprintf(stderr, "ds4-slice-bench: choose exactly one of --capture or --replay\n"); return -1; }
    if (c->capture_file && !c->prompt_file) { fprintf(stderr, "ds4-slice-bench: --capture requires --prompt-file\n"); return -1; }
    if (c->replay_file && c->prompt_file) { fprintf(stderr, "ds4-slice-bench: --replay does not accept --prompt-file\n"); return -1; }
    if (c->replay_file && c->decode_token_set) { fprintf(stderr, "ds4-slice-bench: --decode-token is capture-only\n"); return -1; }
    return 1;
}
static char *read_file(const char *path) {
    FILE *fp = fopen(path, "rb"); if (!fp) { fprintf(stderr, "ds4-slice-bench: open %s: %s\n", path, strerror(errno)); return NULL; }
    if (fseek(fp, 0, SEEK_END) != 0) { fclose(fp); return NULL; }
    long n = ftell(fp); if (n < 0 || (uint64_t)n > SIZE_MAX - 1u) { fclose(fp); return NULL; }
    rewind(fp); char *p = malloc((size_t)n + 1u);
    if (!p || fread(p, 1, (size_t)n, fp) != (size_t)n) { free(p); fclose(fp); return NULL; }
    p[n] = '\0'; fclose(fp); return p;
}
static int mul_size(uint64_t n, uint64_t e, size_t *out) {
    if ((e && n > UINT64_MAX / e) || n * e > SIZE_MAX) return 0;
    *out = (size_t)(n * e); return 1;
}
static int mul_u64(uint64_t a, uint64_t b, uint64_t *out) {
    if (b && a > UINT64_MAX / b) return 0;
    *out = a * b;
    return 1;
}
static int range_ok(uint64_t off, uint64_t bytes, uint64_t file_bytes) {
    return off <= file_bytes && bytes <= file_bytes - off;
}
static int exact_write(FILE *fp, const void *p, size_t n) { return !n || fwrite(p, 1, n, fp) == n; }
static int exact_read(FILE *fp, void *p, size_t n) { return !n || fread(p, 1, n, fp) == n; }
static void corpus_free(corpus *c) {
    if (!c) return;
    free(c->tokens); free(c->chunks); free(c->prefill_hc); free(c->decode_hc);
    memset(c, 0, sizeof(*c));
}
static int corpus_write(const char *path, const corpus *c) {
    size_t cb, tb, pb, db;
    if (!mul_size(c->h.chunk_count, sizeof(*c->chunks), &cb) ||
        !mul_size(c->h.prompt_tokens, sizeof(*c->tokens), &tb) ||
        !mul_size(c->h.prompt_tokens * c->h.hidden_f32, sizeof(float), &pb) ||
        !mul_size(c->h.hidden_f32, sizeof(float), &db)) return fail("write corpus", "size overflow");
    size_t n = strlen(path) + 40; char *tmp = malloc(n);
    if (!tmp) return fail("write corpus", "temporary path allocation failed");
    snprintf(tmp, n, "%s.tmp.%ld", path, (long)getpid());
    FILE *fp = fopen(tmp, "wbx");
    if (!fp) { fprintf(stderr, "ds4-slice-bench: create %s: %s\n", tmp, strerror(errno)); free(tmp); return 0; }
    int ok = exact_write(fp, &c->h, sizeof(c->h)) && exact_write(fp, c->chunks, cb) && exact_write(fp, c->tokens, tb) && exact_write(fp, c->prefill_hc, pb) && exact_write(fp, c->decode_hc, db);
    if (ok && fflush(fp) != 0) ok = 0;
    if (ok && fsync(fileno(fp)) != 0) ok = 0;
    if (fclose(fp) != 0) ok = 0;
    if (ok && rename(tmp, path) != 0) ok = 0;
    if (!ok) { fprintf(stderr, "ds4-slice-bench: atomic corpus write failed: %s\n", strerror(errno)); unlink(tmp); }
    free(tmp); return ok;
}
static int corpus_load(const char *path, corpus *c) {
    memset(c, 0, sizeof(*c)); struct stat st;
    if (stat(path, &st) != 0) { fprintf(stderr, "ds4-slice-bench: stat corpus %s: %s\n", path, strerror(errno)); return 0; }
    FILE *fp = fopen(path, "rb"); if (!fp) return fail("read corpus", strerror(errno));
    if (st.st_size < (off_t)sizeof(c->h) || !exact_read(fp, &c->h, sizeof(c->h))) { fclose(fp); return fail("read corpus", "truncated header"); }
    corpus_header *h = &c->h; uint64_t bytes = (uint64_t)st.st_size; size_t cb, tb, pb, db;
    uint64_t prefill_values = 0;
    int ok = h->magic == CORPUS_MAGIC && h->version == CORPUS_VERSION && h->header_bytes == sizeof(*h) && h->file_bytes == bytes && h->prompt_tokens > 0 && h->chunk_count > 0 && h->chunk_size > 0 && h->chunk_size <= h->ctx_size && h->hidden_f32 > 0 && h->ctx_size > h->prompt_tokens && mul_u64(h->prompt_tokens, h->hidden_f32, &prefill_values) && mul_size(h->chunk_count, sizeof(*c->chunks), &cb) && mul_size(h->prompt_tokens, sizeof(*c->tokens), &tb) && mul_size(prefill_values, sizeof(float), &pb) && mul_size(h->hidden_f32, sizeof(float), &db) && range_ok(h->chunk_offset, cb, bytes) && range_ok(h->token_offset, tb, bytes) && range_ok(h->prefill_hc_offset, pb, bytes) && range_ok(h->decode_hc_offset, db, bytes);
    if (ok && (h->chunk_offset != sizeof(*h) || h->token_offset != h->chunk_offset + cb || h->prefill_hc_offset != h->token_offset + tb || h->decode_hc_offset != h->prefill_hc_offset + pb || h->decode_hc_offset + db != bytes)) ok = 0;
    if (!ok) { fclose(fp); return fail("read corpus", "invalid or incompatible header"); }
    c->tokens = malloc(tb); c->chunks = malloc(cb); c->prefill_hc = malloc(pb); c->decode_hc = malloc(db);
    ok = c->tokens && c->chunks && c->prefill_hc && c->decode_hc;
    if (ok && (fseek(fp, (long)h->chunk_offset, SEEK_SET) || !exact_read(fp, c->chunks, cb))) ok = 0;
    if (ok && (fseek(fp, (long)h->token_offset, SEEK_SET) || !exact_read(fp, c->tokens, tb))) ok = 0;
    if (ok && (fseek(fp, (long)h->prefill_hc_offset, SEEK_SET) || !exact_read(fp, c->prefill_hc, pb))) ok = 0;
    if (ok && (fseek(fp, (long)h->decode_hc_offset, SEEK_SET) || !exact_read(fp, c->decode_hc, db))) ok = 0;
    fclose(fp); if (!ok) { corpus_free(c); return fail("read corpus", "truncated payload"); }
    uint64_t hash = UINT64_C(1469598103934665603); hash = hash_bytes(hash, c->tokens, tb); hash = hash_bytes(hash, c->chunks, cb); hash = hash_bytes(hash, c->prefill_hc, pb); hash = hash_bytes(hash, c->decode_hc, db);
    if (hash != h->payload_hash) { corpus_free(c); return fail("read corpus", "payload checksum mismatch"); }
    uint64_t pos = 0;
    for (uint32_t i = 0; i < h->chunk_count; i++) { corpus_chunk *ch = &c->chunks[i]; if (ch->pos0 != pos || ch->n_tokens == 0 || ch->n_tokens > h->chunk_size || ch->hc_row_offset != pos) { corpus_free(c); return fail("read corpus", "invalid chunk metadata"); } pos += ch->n_tokens; }
    if (pos != h->prompt_tokens) { corpus_free(c); return fail("read corpus", "chunk metadata does not cover prompt"); }
    return 1;
}
static int open_engine(const config *c, int ctx, uint32_t first, uint32_t last, ds4_engine **out) {
    ds4_engine_options o; memset(&o, 0, sizeof(o)); o.model_path = c->model; o.backend = DS4_BACKEND_CUDA; o.context_size = ctx; o.prefill_chunk = c->chunk; o.warm_weights = true; o.load_slice = true; o.load_layer_start = first; o.load_layer_end = last; o.load_output = false;
    return ds4_engine_open(out, &o) == 0 && *out;
}
static int capture(const config *c) {
    ds4_engine *e = NULL; if (!open_engine(c, c->ctx, 0, SOURCE_END, &e)) return 2;
    if (ds4_engine_layer_count(e) <= SLICE_END) { ds4_engine_close(e); fail("capture", "model has fewer than 8 executable layers"); return 2; }
    char *text = read_file(c->prompt_file); if (!text) { ds4_engine_close(e); return 2; }
    ds4_tokens prompt = {0}; ds4_tokenize_rendered_chat(e, text, &prompt); free(text);
    if (prompt.len <= 0 || prompt.len >= c->ctx) { ds4_tokens_free(&prompt); ds4_engine_close(e); fail("capture", "prompt is empty or exceeds context"); return 2; }
    int token = c->decode_token_set ? c->decode_token : prompt.v[prompt.len - 1];
    if (token < 0 || token >= ds4_engine_vocab_size(e)) { ds4_tokens_free(&prompt); ds4_engine_close(e); fail("capture", "decode token is outside vocabulary"); return 2; }
    ds4_session *s = NULL; if (ds4_session_create(&s, e, c->ctx) != 0) { ds4_tokens_free(&prompt); ds4_engine_close(e); return 2; }
    uint64_t hidden = ds4_engine_hidden_f32_values(e); uint32_t chunk = c->chunk ? c->chunk : (uint32_t)ds4_session_prefill_cap(s); if (chunk > (uint32_t)prompt.len) chunk = (uint32_t)prompt.len;
    uint32_t nchunks = ((uint32_t)prompt.len + chunk - 1u) / chunk; size_t cb, tb, pb, db;
    uint64_t prefill_values = 0;
    if (!mul_u64((uint64_t)prompt.len, hidden, &prefill_values) || !mul_size(nchunks, sizeof(corpus_chunk), &cb) || !mul_size(prompt.len, sizeof(int32_t), &tb) || !mul_size(prefill_values, sizeof(float), &pb) || !mul_size(hidden, sizeof(float), &db)) { ds4_session_free(s); ds4_tokens_free(&prompt); ds4_engine_close(e); fail("capture", "corpus size overflow"); return 2; }
    corpus cpr = {0}; cpr.h = (corpus_header){ .magic = CORPUS_MAGIC, .version = CORPUS_VERSION, .header_bytes = sizeof(cpr.h), .model_id = (uint32_t)ds4_engine_model_id(e), .model_name_hash = hash_text(ds4_engine_model_name(e)), .model_bytes = ds4_engine_model_bytes(e), .hidden_f32 = hidden, .ctx_size = (uint32_t)c->ctx, .chunk_size = chunk, .prompt_tokens = (uint32_t)prompt.len, .chunk_count = nchunks, .decode_token = token, .chunk_offset = sizeof(cpr.h), .token_offset = sizeof(cpr.h) + cb, .prefill_hc_offset = sizeof(cpr.h) + cb + tb, .decode_hc_offset = sizeof(cpr.h) + cb + tb + pb, .file_bytes = sizeof(cpr.h) + cb + tb + pb + db };
    cpr.tokens = malloc(tb); cpr.chunks = calloc(nchunks, sizeof(*cpr.chunks)); cpr.prefill_hc = calloc(1, pb); cpr.decode_hc = calloc(1, db);
    if (!cpr.tokens || !cpr.chunks || !cpr.prefill_hc || !cpr.decode_hc) { corpus_free(&cpr); ds4_session_free(s); ds4_tokens_free(&prompt); ds4_engine_close(e); fail("capture", "corpus allocation failed"); return 2; }
    for (int i = 0; i < prompt.len; i++) cpr.tokens[i] = prompt.v[i];
    char err[512] = {0}; uint32_t pos = 0;
    for (uint32_t i = 0; i < nchunks; i++) { uint32_t n = (uint32_t)prompt.len - pos; if (n > chunk) n = chunk; cpr.chunks[i] = (corpus_chunk){ pos, n, pos }; if (ds4_session_eval_layer_slice(s, prompt.v + pos, n, pos, 0, SOURCE_END, NULL, cpr.prefill_hc + (uint64_t)pos * hidden, false, NULL, err, sizeof(err)) != 0) { fail("capture source 0:3", err); corpus_free(&cpr); ds4_session_free(s); ds4_tokens_free(&prompt); ds4_engine_close(e); return 2; } pos += n; }
    if (ds4_session_eval_layer_slice(s, &token, 1, (uint32_t)prompt.len, 0, SOURCE_END, NULL, cpr.decode_hc, false, NULL, err, sizeof(err)) != 0) { fail("capture source decode 0:3", err); corpus_free(&cpr); ds4_session_free(s); ds4_tokens_free(&prompt); ds4_engine_close(e); return 2; }
    uint64_t hash = UINT64_C(1469598103934665603); hash = hash_bytes(hash, cpr.tokens, tb); hash = hash_bytes(hash, cpr.chunks, cb); hash = hash_bytes(hash, cpr.prefill_hc, pb); hash = hash_bytes(hash, cpr.decode_hc, db); cpr.h.payload_hash = hash;
    int ok = corpus_write(c->capture_file, &cpr); if (ok) printf("slice_capture layers=0:%u prompt_tokens=%u chunk=%u chunks=%u hidden_f32=%" PRIu64 " corpus=%s payload_hash=%016" PRIx64 "\n", SOURCE_END, cpr.h.prompt_tokens, chunk, nchunks, hidden, c->capture_file, hash);
    corpus_free(&cpr); ds4_session_free(s); ds4_tokens_free(&prompt); ds4_engine_close(e); return ok ? 0 : 2;
}
static int reset_target(ds4_session *s) { char err[256] = {0}; return ds4_session_layer_slice_reset(s, err, sizeof(err)) == 0 || (fail("reset target", err), 0); }
static int target_prefill(ds4_session *s, const corpus *c, uint64_t hidden, float *out, uint64_t *hash, metrics *m) {
    char err[512] = {0}; double t0 = now_ms(); unsetenv("DS4_VULKAN_TIMELINE_LAYER");
    for (uint32_t i = 0; i < c->h.chunk_count; i++) { corpus_chunk *ch = &c->chunks[i]; bool last = i + 1u == c->h.chunk_count; if (ds4_session_eval_layer_slice(s, (const int *)c->tokens + ch->pos0, ch->n_tokens, ch->pos0, SLICE_START, SLICE_END, c->prefill_hc + ch->hc_row_offset * hidden, last ? out : NULL, false, NULL, err, sizeof(err)) != 0) return fail("replay target prefill 4:7", err); if (last && hash) *hash = hash_f32(out + (uint64_t)(ch->n_tokens - 1u) * hidden, hidden); }
    if (m) m->prefill_wall_ms += now_ms() - t0;
    return 1;
}
static int target_decode(ds4_session *s, const corpus *c, uint64_t hidden, float *out, uint64_t *hash, metrics *m) {
    char err[512] = {0}; double t0 = now_ms(); setenv("DS4_VULKAN_TIMELINE_LAYER", "4", 1); if (!ds4_gpu_worker_slice_begin(SLICE_START, SLICE_END)) return fail("decode worker slice begin", "worker slice unavailable");
    int rc = ds4_session_eval_layer_slice(s, &c->h.decode_token, 1, c->h.prompt_tokens, SLICE_START, SLICE_END, c->decode_hc, out, false, NULL, err, sizeof(err)); int end = ds4_gpu_worker_slice_end(SLICE_START, SLICE_END); if (rc != 0) return fail("decode target 4:7", err); if (!end) return fail("decode worker slice end", "retirement failed"); if (hash) *hash = hash_f32(out, hidden);
    if (m) { ds4_gpu_timeline_stats x = {0}; ds4_gpu_timeline_layer_read_stats(&x); m->decode_wall_ms += now_ms() - t0; m->gpu_ms += x.gpu_ms; m->submit_cpu_ms += x.submit_cpu_ms; m->wait_cpu_ms += x.wait_cpu_ms; m->dispatches += x.dispatches; m->barriers += x.barriers; m->submissions += x.submissions; m->waits += x.waits; } return 1;
}
static int replay(const config *c) {
    corpus cpr; if (!corpus_load(c->replay_file, &cpr)) return 2; if (c->ctx_set && (uint32_t)c->ctx != cpr.h.ctx_size) { corpus_free(&cpr); fail("replay", "--ctx does not match corpus"); return 2; } if (c->chunk_set && c->chunk != cpr.h.chunk_size) { corpus_free(&cpr); fail("replay", "--chunk does not match corpus"); return 2; }
    config open = *c; open.chunk = cpr.h.chunk_size; ds4_engine *e = NULL; if (!open_engine(&open, (int)cpr.h.ctx_size, SLICE_START, SLICE_END, &e)) { corpus_free(&cpr); return 2; }
    if ((uint32_t)ds4_engine_model_id(e) != cpr.h.model_id || hash_text(ds4_engine_model_name(e)) != cpr.h.model_name_hash || ds4_engine_model_bytes(e) != cpr.h.model_bytes || ds4_engine_hidden_f32_values(e) != cpr.h.hidden_f32 || ds4_engine_layer_count(e) <= SLICE_END || cpr.h.decode_token < 0 || cpr.h.decode_token >= ds4_engine_vocab_size(e)) { fprintf(stderr, "ds4-slice-bench: corpus model/shape does not match replay engine\n"); corpus_free(&cpr); ds4_engine_close(e); return 2; }
    ds4_session *s = NULL; if (ds4_session_create(&s, e, cpr.h.ctx_size) != 0) { corpus_free(&cpr); ds4_engine_close(e); return 2; } size_t out_bytes; uint64_t out_values = 0; if (!mul_u64(cpr.h.chunk_size, cpr.h.hidden_f32, &out_values) || !mul_size(out_values, sizeof(float), &out_bytes)) { ds4_session_free(s); corpus_free(&cpr); ds4_engine_close(e); return 2; } float *out = calloc(1, out_bytes); if (!out) { ds4_session_free(s); corpus_free(&cpr); ds4_engine_close(e); return 2; }
    setenv("DS4_VULKAN_WORKER_SLICE_BATCH", "1", 1); setenv("DS4_VULKAN_SLICE_BENCH", "1", 1); setenv("DS4_VULKAN_TIMELINE_COUNT", "1024", 0); setenv("DS4_VULKAN_TIMELINE_LAYER", "4", 1);
    uint64_t base_p = 0, base_d = 0; if (!target_prefill(s, &cpr, cpr.h.hidden_f32, out, &base_p, NULL) || !target_decode(s, &cpr, cpr.h.hidden_f32, out, &base_d, NULL) || !reset_target(s)) goto replay_fail;
    for (int i = 0; i < c->warmup; i++) if (!target_prefill(s, &cpr, cpr.h.hidden_f32, out, NULL, NULL) || !target_decode(s, &cpr, cpr.h.hidden_f32, out, NULL, NULL) || !reset_target(s)) goto replay_fail;
    metrics m = {0}; uint64_t p = 0, d = 0; for (int i = 0; i < c->iters; i++) { if (!target_prefill(s, &cpr, cpr.h.hidden_f32, out, &p, &m) || !target_decode(s, &cpr, cpr.h.hidden_f32, out, &d, &m) || !reset_target(s)) goto replay_fail; if (p != base_p || d != base_d) { fprintf(stderr, "ds4-slice-bench: output hash changed at iteration %d (prefill=%016" PRIx64 "/%016" PRIx64 " decode=%016" PRIx64 "/%016" PRIx64 ")\n", i, p, base_p, d, base_d); free(out); ds4_session_free(s); corpus_free(&cpr); ds4_engine_close(e); return 3; } }
    { double pm = m.prefill_wall_ms / c->iters, dm = m.decode_wall_ms / c->iters; printf("slice_bench layers=%u:%u target_only=1 prompt_tokens=%u chunk=%u chunks=%u warmup=%d iters=%d decode_token=%d corpus=%s\n", SLICE_START, SLICE_END, cpr.h.prompt_tokens, cpr.h.chunk_size, cpr.h.chunk_count, c->warmup, c->iters, cpr.h.decode_token, c->replay_file); printf("corpus_hash payload=%016" PRIx64 "\noutput_hash baseline_prefill=%016" PRIx64 " baseline_decode=%016" PRIx64 "\n", cpr.h.payload_hash, base_p, base_d); printf("timing prefill_wall_ms=%.3f prefill_tps=%.3f decode_wall_ms=%.3f decode_tps=%.3f total_wall_ms=%.3f\n", pm, cpr.h.prompt_tokens / (pm / 1000.0), dm, 1000.0 / dm, pm + dm); printf("decode_gpu_ms=%.3f submit_cpu_ms=%.3f wait_cpu_ms=%.3f dispatches=%.1f barriers=%.1f submissions=%.1f waits=%.1f\n", m.gpu_ms / c->iters, m.submit_cpu_ms / c->iters, m.wait_cpu_ms / c->iters, (double)m.dispatches / c->iters, (double)m.barriers / c->iters, (double)m.submissions / c->iters, (double)m.waits / c->iters); printf("hash_check=PASS\n"); }
    free(out); ds4_session_free(s); corpus_free(&cpr); ds4_engine_close(e); return 0;
replay_fail:
    free(out); ds4_session_free(s); corpus_free(&cpr); ds4_engine_close(e); return 2;
}
int main(int argc, char **argv) { config c; int p = parse_args(argc, argv, &c); if (p <= 0) return p == 0 ? 0 : 2; return c.capture_file ? capture(&c) : replay(&c); }
