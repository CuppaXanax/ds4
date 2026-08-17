/* Benchmark A driver: production Flash weight-span ledger + resident stream.
 *
 * This intentionally parses only GGUF metadata.  It never decodes a tensor or
 * constructs activations: the backend receives the exact file ranges that the
 * decode kernels would map for one layer and the chosen top-6 experts.
 * Linux/BC-250 is the supported runtime; the source remains ordinary C++17 so
 * it can be built beside the existing Vulkan backend.
 */
#include <vulkan/vulkan.h>
#include "../include/vk_mem_alloc.h"
#include "../../ds4_vulkan.h"
#include "../../ds4_gpu.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <glob.h>
#include <map>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

namespace {

struct Reader {
    const uint8_t *p;
    const uint8_t *end;
    bool ok = true;
    template <typename T> T get() {
        if (!ok || (size_t)(end - p) < sizeof(T)) { ok = false; return T{}; }
        T v; memcpy(&v, p, sizeof(v)); p += sizeof(v); return v;
    }
    std::string str() {
        const uint64_t n = get<uint64_t>();
        if (!ok || n > (uint64_t)(end - p)) { ok = false; return {}; }
        std::string s((const char *)p, (size_t)n); p += n; return s;
    }
    void bytes(uint64_t n) {
        if (!ok || n > (uint64_t)(end - p)) { ok = false; return; }
        p += n;
    }
};

enum : uint32_t {
    GGUF_UINT8 = 0, GGUF_INT8 = 1, GGUF_UINT16 = 2, GGUF_INT16 = 3,
    GGUF_UINT32 = 4, GGUF_INT32 = 5, GGUF_FLOAT32 = 6, GGUF_BOOL = 7,
    GGUF_STRING = 8, GGUF_ARRAY = 9, GGUF_UINT64 = 10, GGUF_INT64 = 11,
    GGUF_FLOAT64 = 12,
};

static bool skip_value(Reader &r, uint32_t type) {
    switch (type) {
    case GGUF_UINT8: case GGUF_INT8: case GGUF_BOOL: r.bytes(1); return r.ok;
    case GGUF_UINT16: case GGUF_INT16: r.bytes(2); return r.ok;
    case GGUF_UINT32: case GGUF_INT32: case GGUF_FLOAT32: r.bytes(4); return r.ok;
    case GGUF_UINT64: case GGUF_INT64: case GGUF_FLOAT64: r.bytes(8); return r.ok;
    case GGUF_STRING: (void)r.str(); return r.ok;
    case GGUF_ARRAY: {
        const uint32_t elem_type = r.get<uint32_t>();
        const uint64_t count = r.get<uint64_t>();
        if (!r.ok || count > (uint64_t)(r.end - r.p)) return r.ok = false;
        for (uint64_t i = 0; i < count; i++) if (!skip_value(r, elem_type)) return false;
        return true;
    }
    default: r.ok = false; return false;
    }
}

struct Tensor {
    std::string name;
    uint64_t dim[4] = {};
    uint32_t ndim = 0;
    uint32_t type = 0;
    uint64_t relative_offset = 0;
    uint64_t bytes = 0;
};

struct Gguf {
    std::vector<Tensor> tensors;
    uint64_t data_offset = 0;
    uint64_t alignment = 32;
};

static bool parse_gguf(const uint8_t *map, size_t size, Gguf &out) {
    if (size < 32) return false;
    Reader r{map, map + size};
    if (r.get<uint32_t>() != 0x46554747u) return false; /* GGUF */
    const uint32_t version = r.get<uint32_t>();
    if (version < 1 || version > 3) return false;
    const uint64_t tensor_count = r.get<uint64_t>();
    const uint64_t kv_count = r.get<uint64_t>();
    if (!r.ok || tensor_count > 1000000 || kv_count > 1000000) return false;
    for (uint64_t i = 0; i < kv_count; i++) {
        const std::string key = r.str();
        (void)key;
        const uint32_t type = r.get<uint32_t>();
        if (!r.ok) return false;
        if (key == "general.alignment" && type == GGUF_UINT32)
            out.alignment = r.get<uint32_t>();
        else if (!skip_value(r, type)) return false;
    }
    out.tensors.reserve((size_t)tensor_count);
    for (uint64_t i = 0; i < tensor_count; i++) {
        Tensor t;
        t.name = r.str();
        t.ndim = r.get<uint32_t>();
        if (!r.ok || t.ndim == 0 || t.ndim > 4) return false;
        for (uint32_t d = 0; d < t.ndim; d++) t.dim[d] = r.get<uint64_t>();
        t.type = r.get<uint32_t>();
        t.relative_offset = r.get<uint64_t>();
        if (!r.ok) return false;
        out.tensors.push_back(std::move(t));
    }
    const uint64_t pos = (uint64_t)(r.p - map);
    out.data_offset = (pos + out.alignment - 1) / out.alignment * out.alignment;
    return out.data_offset <= size;
}

static uint64_t tensor_bytes(Tensor &t) {
    uint64_t block_elems = 1, block_bytes = 0;
    switch (t.type) {
    case 0:  block_elems = 1;   block_bytes = 4;   break; /* F32 */
    case 1:  block_elems = 1;   block_bytes = 2;   break; /* F16 */
    case 2:  block_elems = 32;  block_bytes = 18;  break; /* Q4_0 */
    case 3:  block_elems = 32;  block_bytes = 20;  break; /* Q4_1 */
    case 6:  block_elems = 32;  block_bytes = 22;  break; /* Q5_0 */
    case 7:  block_elems = 32;  block_bytes = 24;  break; /* Q5_1 */
    case 8:  block_elems = 32;  block_bytes = 34; break; /* Q8_0 */
    case 9:  block_elems = 32;  block_bytes = 36;  break; /* Q8_1 */
    case 10: block_elems = 256; block_bytes = 84;  break; /* Q2_K */
    case 11: block_elems = 256; block_bytes = 110; break; /* Q3_K */
    case 12: block_elems = 256; block_bytes = 144; break; /* Q4_K */
    case 13: block_elems = 256; block_bytes = 176; break; /* Q5_K */
    case 14: block_elems = 256; block_bytes = 210; break; /* Q6_K */
    case 15: block_elems = 256; block_bytes = 292; break; /* Q8_K */
    case 16: block_elems = 256; block_bytes = 66;  break; /* IQ2_XXS */
    case 17: block_elems = 256; block_bytes = 74;  break; /* IQ2_XS */
    case 18: block_elems = 256; block_bytes = 98;  break; /* IQ3_XXS */
    case 19: block_elems = 256; block_bytes = 50;  break; /* IQ1_S */
    case 20: block_elems = 32;  block_bytes = 18;  break; /* IQ4_NL */
    case 21: block_elems = 256; block_bytes = 110; break; /* IQ3_S */
    case 22: block_elems = 256; block_bytes = 82;  break; /* IQ2_S */
    case 23: block_elems = 256; block_bytes = 136; break; /* IQ4_XS */
    case 24: block_elems = 1;   block_bytes = 1;   break; /* I8 */
    case 25: block_elems = 1;   block_bytes = 2;   break; /* I16 */
    case 26: block_elems = 1;   block_bytes = 4;   break; /* I32 */
    case 27: block_elems = 1;   block_bytes = 8;   break; /* I64 */
    case 28: block_elems = 1;   block_bytes = 8;   break; /* F64 */
    case 29: block_elems = 256; block_bytes = 56;  break; /* IQ1_M */
    case 30: block_elems = 1;   block_bytes = 2;   break; /* BF16 */
    case 39: block_elems = 32;  block_bytes = 17;  break; /* MXFP4 */
    case 40: block_elems = 64;  block_bytes = 18;  break; /* NVFP4 */
    case 41: block_elems = 256; block_bytes = 36;  break; /* Q1_0 */
    default: return 0;
    }
    if (t.dim[0] == 0 || t.dim[0] % block_elems) return 0;
    uint64_t rows = 1;
    for (uint32_t i = 1; i < t.ndim; i++) {
        if (t.dim[i] && rows > UINT64_MAX / t.dim[i]) return 0;
        rows *= t.dim[i];
    }
    const uint64_t row_bytes = (t.dim[0] / block_elems) * block_bytes;
    if (rows > UINT64_MAX / row_bytes) return 0;
    t.bytes = rows * row_bytes;
    return t.bytes;
}

static const Tensor *find_tensor(const Gguf &g, const std::string &name) {
    for (const Tensor &t : g.tensors) if (t.name == name) return &t;
    return nullptr;
}

static std::string arg_value(int &i, int argc, char **argv) {
    return i + 1 < argc ? argv[++i] : std::string();
}

static std::vector<uint32_t> parse_ids(const std::string &s) {
    std::vector<uint32_t> ids;
    size_t at = 0;
    while (at < s.size()) {
        size_t comma = s.find(',', at);
        const std::string part = s.substr(at, comma == std::string::npos ? comma : comma - at);
        char *end = nullptr; unsigned long v = strtoul(part.c_str(), &end, 10);
        if (!part.empty() && end && *end == '\0') ids.push_back((uint32_t)v);
        if (comma == std::string::npos) break;
        at = comma + 1;
    }
    return ids;
}

static void usage(const char *prog) {
    fprintf(stderr,
        "usage: %s MODEL [--layer N] [--experts 0,1,2,3,4,5] "
        "[--repeats N] [--dense NAME] [--ledger-only]\n", prog);
}

static std::string sysfs_first_line(const char *pattern) {
    glob_t matches{};
    if (glob(pattern, 0, nullptr, &matches) != 0 || matches.gl_pathc == 0) {
        globfree(&matches); return {};
    }
    std::ifstream f(matches.gl_pathv[0]);
    std::string line;
    std::getline(f, line);
    globfree(&matches);
    return line;
}

static void print_device_telemetry(const char *when) {
    const std::string clock = sysfs_first_line("/sys/class/drm/card*/device/pp_dpm_sclk");
    const std::string temp = sysfs_first_line("/sys/class/drm/card*/device/hwmon/hwmon*/temp1_input");
    if (!clock.empty() || !temp.empty())
        printf("device telemetry %s: clock=%s temp_mC=%s\n", when,
               clock.empty() ? "unavailable" : clock.c_str(),
               temp.empty() ? "unavailable" : temp.c_str());
    else
        printf("device telemetry %s: unavailable\n", when);
}

struct Ledger {
    uint64_t dense = 0, gate_up = 0, down = 0;
    uint64_t attention = 0, shared = 0, router_indexer = 0, mhc = 0, output = 0, other = 0;
    void add_nonexpert(const std::string &name, uint64_t bytes) {
        dense += bytes;
        if (name.find("attn_") != std::string::npos) attention += bytes;
        else if (name.find("shexp") != std::string::npos) shared += bytes;
        else if (name.find("indexer") != std::string::npos ||
                 name.find("gate_inp") != std::string::npos ||
                 name.find("exp_probs") != std::string::npos ||
                 name.find("router") != std::string::npos) router_indexer += bytes;
        else if (name.find("hc_") != std::string::npos) mhc += bytes;
        else if (name.find("output") != std::string::npos) output += bytes;
        else other += bytes;
    }
    void print() const {
        const uint64_t distinct = dense + gate_up + down;
        printf("ledger theoretical distinct minimum: layer_nonexpert=%" PRIu64 " iq2_gate_up=%" PRIu64
               " q2_down=%" PRIu64 " total=%" PRIu64 "\n",
               dense, gate_up, down, distinct);
        printf("ledger layer breakdown: attention=%" PRIu64 " shared_experts=%" PRIu64
               " router_indexer=%" PRIu64 " mHC=%" PRIu64 " output=%" PRIu64
               " norm/other=%" PRIu64 "\n", attention, shared, router_indexer,
               mhc, output, other);
        printf("ledger Benchmark-A logical span requests: layer_nonexpert=%" PRIu64 " iq2_gate_up=%" PRIu64
               " q2_down=%" PRIu64 " total=%" PRIu64 " rereads=0\n",
               dense, gate_up, down, distinct);
        printf("ledger note: production rereads are not inferred; repeats are timed passes, not denominator bytes\n");
        printf("ledger activation/KV bytes: excluded (0)\n");
    }
};

} /* namespace */

int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 2; }
    const char *model_path = argv[1];
    int layer = -1;
    uint32_t repeats = 20;
    bool ledger_only = false;
    std::string dense_name;
    std::vector<uint32_t> experts = {0, 1, 2, 3, 4, 5};
    for (int i = 2; i < argc; i++) {
        const std::string arg = argv[i];
        if (arg == "--layer") layer = atoi(arg_value(i, argc, argv).c_str());
        else if (arg == "--experts") experts = parse_ids(arg_value(i, argc, argv));
        else if (arg == "--repeats") repeats = (uint32_t)strtoul(arg_value(i, argc, argv).c_str(), nullptr, 10);
        else if (arg == "--dense") dense_name = arg_value(i, argc, argv);
        else if (arg == "--ledger-only") ledger_only = true;
        else { usage(argv[0]); return 2; }
    }
    if (experts.empty() || repeats == 0) { usage(argv[0]); return 2; }
    const int fd = open(model_path, O_RDONLY);
    if (fd < 0) { perror("open model"); return 1; }
    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) { perror("fstat model"); close(fd); return 1; }
    const size_t model_size = (size_t)st.st_size;
    void *map = mmap(nullptr, model_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (map == MAP_FAILED) { perror("mmap model"); return 1; }
    Gguf gguf;
    if (!parse_gguf((const uint8_t *)map, model_size, gguf)) {
        fprintf(stderr, "roofline A: invalid/unsupported GGUF metadata\n");
        munmap(map, model_size); return 1;
    }
    for (Tensor &t : gguf.tensors) tensor_bytes(t);

    auto routed_for_layer = [&](int il, const char *suffix) -> const Tensor * {
        char name[128]; snprintf(name, sizeof(name), "blk.%d.%s", il, suffix);
        return find_tensor(gguf, name);
    };
    if (layer < 0) {
        for (int il = 0; il < 256; il++) {
            const Tensor *g = routed_for_layer(il, "ffn_gate_exps.weight");
            const Tensor *u = routed_for_layer(il, "ffn_up_exps.weight");
            const Tensor *d = routed_for_layer(il, "ffn_down_exps.weight");
            if (g && u && d && g->type == 16 && u->type == 16 && d->type == 10) {
                layer = il; break;
            }
        }
    }
    if (layer < 0) {
        fprintf(stderr, "roofline A: no Flash IQ2 gate/up + Q2 down layer found\n");
        munmap(map, model_size); return 1;
    }
    const Tensor *gate = routed_for_layer(layer, "ffn_gate_exps.weight");
    const Tensor *up = routed_for_layer(layer, "ffn_up_exps.weight");
    const Tensor *down = routed_for_layer(layer, "ffn_down_exps.weight");
    if (!gate || !up || !down || gate->type != 16 || up->type != 16 || down->type != 10 ||
        gate->ndim != 3 || up->ndim != 3 || down->ndim != 3) {
        fprintf(stderr, "roofline A: selected layer %d is not IQ2/IQ2/Q2 Flash\n", layer);
        munmap(map, model_size); return 1;
    }
    for (uint32_t e : experts) {
        if (e >= gate->dim[2] || e >= up->dim[2] || e >= down->dim[2]) {
            fprintf(stderr, "roofline A: expert id %u exceeds tensor expert count\n", e);
            munmap(map, model_size); return 1;
        }
    }
    std::vector<ds4_vulkan_roofline_span> spans;
    std::vector<std::string> labels;
    Ledger ledger;
    auto add_span = [&](const Tensor *t, uint32_t family, uint32_t expert, const char *label) {
        if (!t || t->bytes == 0) return false;
        if (family != 0 && (t->ndim < 3 || t->dim[2] == 0)) return false;
        const uint64_t expert_bytes = family == 0 ? t->bytes : t->bytes / t->dim[2];
        const uint64_t offset = gguf.data_offset + t->relative_offset + expert * expert_bytes;
        if (offset > model_size || expert_bytes > model_size - offset || (expert_bytes & 3u)) return false;
        spans.push_back({offset, expert_bytes, family, (uint32_t)layer, expert});
        char text[160]; snprintf(text, sizeof(text), "%s expert=%u offset=%" PRIu64 " bytes=%" PRIu64,
                                  label, expert, offset, expert_bytes);
        labels.emplace_back(text);
        if (family == 0) ledger.add_nonexpert(label, expert_bytes);
        else if (family == 1) ledger.gate_up += expert_bytes;
        else ledger.down += expert_bytes;
        return true;
    };
    /* Dense attention/shared projections are visited before the routed MLP.
     * The routed order is all selected gate rows, all selected up rows, then
     * all selected down rows, matching streamed dispatch mappings. */
    if (!dense_name.empty()) {
        const Tensor *t = find_tensor(gguf, dense_name);
        if (!t || t->type != 8 || !add_span(t, 0, 0, "dense")) {
            fprintf(stderr, "roofline A: --dense must name a Q8_0 tensor\n"); return 1;
        }
    } else {
        /* Include every non-expert layer weight that the production layer
         * map can bind: attention, HC, norms, router/indexer, and shared
         * expert projections.  Only the three routed stacks are replaced by
         * selected-expert ranges below. */
        char prefix[32]; snprintf(prefix, sizeof(prefix), "blk.%d.", layer);
        for (const Tensor &t : gguf.tensors) {
            const bool is_weight = t.name.size() >= 7 &&
                t.name.compare(t.name.size() - 7, 7, ".weight") == 0;
            const bool is_bias = t.name.size() >= 5 &&
                t.name.compare(t.name.size() - 5, 5, ".bias") == 0;
            if (t.name.compare(0, strlen(prefix), prefix) != 0 || (!is_weight && !is_bias))
                continue;
            if (t.name.find("ffn_gate_exps.weight") != std::string::npos ||
                t.name.find("ffn_up_exps.weight") != std::string::npos ||
                t.name.find("ffn_down_exps.weight") != std::string::npos)
                continue;
            if (!add_span(&t, 0, 0, t.name.c_str())) {
                fprintf(stderr, "roofline A: cannot map layer tensor %s\n", t.name.c_str());
                return 1;
            }
        }
    }
    for (uint32_t e : experts) if (!add_span(gate, 1, e, "iq2_gate")) return 1;
    for (uint32_t e : experts) if (!add_span(up, 1, e, "iq2_up")) return 1;
    for (uint32_t e : experts) if (!add_span(down, 2, e, "q2_down")) return 1;
    if (spans.empty()) { fprintf(stderr, "roofline A: no spans\n"); return 1; }
    printf("roofline A model=%s layer=%d experts=", model_path, layer);
    for (size_t i = 0; i < experts.size(); i++) printf("%s%u", i ? "," : "", experts[i]);
    printf(" repeats=%u spans=%zu\n", repeats, spans.size());
    for (const std::string &label : labels) printf("span %s\n", label.c_str());
    ledger.print();

    if (ledger_only) {
        munmap(map, model_size);
        return 0;
    }

    print_device_telemetry("before");
    if (!ds4_gpu_init()) { fprintf(stderr, "roofline A: Vulkan init failed\n"); munmap(map, model_size); return 1; }
    print_device_telemetry("initialized");
    ds4_vulkan_roofline_result result{};
    const int ok = ds4_vulkan_roofline_weight_stream(map, model_size, spans.data(),
                                                      (uint32_t)spans.size(), repeats, &result);
    if (ok) {
        printf("warm upload bytes=%" PRIu64 " ns=%" PRIu64 "\n",
               result.warm_upload_bytes, result.warm_upload_ns);
        printf("timed useful bytes/pass=%" PRIu64 " dispatches/pass=%u repeats=%u\n",
               result.useful_bytes_per_pass, result.dispatches_per_pass, result.repeats);
        printf("timed family bytes/pass: layer_nonexpert=%" PRIu64 " iq2_gate_up=%" PRIu64
               " q2_down=%" PRIu64 "\n", result.family_bytes_per_pass[0],
               result.family_bytes_per_pass[1], result.family_bytes_per_pass[2]);
        printf("gpu timestamp ns=%" PRIu64 " valid=%u checksum=0x%016" PRIx64
               " effective=%.3f GB/s %.3f GiB/s\n", result.gpu_ns, result.timestamp_valid,
               result.checksum, result.effective_gb_s, result.effective_gib_s);
    } else fprintf(stderr, "roofline A: benchmark failed (timestamp or dispatch)\n");
    print_device_telemetry("after");
    ds4_gpu_cleanup();
    munmap(map, model_size);
    return ok ? 0 : 1;
}
