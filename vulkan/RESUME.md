# DS4 Vulkan backend — RESUME (ripresa sessione)

> **Documento storico.** Queste note descrivono la fase bootstrap Vulkan del
> 2026-08-01 e non sono più la fonte di verità operativa. Per lo stato attuale
> BC-250, i risultati qualificati, gli esperimenti accettati/rifiutati e i
> prossimi obiettivi vedere
> [`../VULKAN_BC250_KERNEL_AUDIT.md`](../VULKAN_BC250_KERNEL_AUDIT.md) e
> [`../VULKAN_BC250_EXPERIMENTS.md`](../VULKAN_BC250_EXPERIMENTS.md).

> Ultimo aggiornamento: 2026-08-01 (fine sessione lunga, compatto in ECA).
> Questo file è la fonte di verità per riprendere il lavoro: riassume tutto lo
> stato, i contratti, i bug già risolti, i risultati dei test e i prossimi passi.
> Leggere anche `AGENT.md` → sezione "Vulkan backend" (contratti dettagliati).

---

## 1. Obiettivo

Aggiungere supporto Vulkan a **DS4** (`/home/nixo/git/ds4`), motore di
inferenza LLM (DeepSeek V4 Flash, 81 GiB, stile llama.cpp) in C. Obiettivo
evoluto: build Vulkan funzionante **solo Vulkan (no fallback CPU)**, kernel
corretti testati, modello che produce output reale.

---

## 2. Contesto macchina (CRITICO)

- Guix; iGPU AMD 890M (Strix Point), driver **RADV mesa 26.0.2**;
  heap Vulkan ~47 GiB (31.5 device + 15.8 host); RAM 93 GiB.
- Modello:
  `gguf/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2.gguf`
  (81 GiB).
- **Gli agenti/subagent NON hanno accesso a `/dev/dri`** (EACCES nel sandbox
  bwrap/cgroup): possono compilare ma NON eseguire GPU.
  **L'utente esegue i test e incolla l'output.** Rispondere in italiano.
- Toolchain: `g++`/`gcc` dal profilo Guix (niente `cc` — usare `CC=gcc`),
  `glslangValidator` su PATH, `-lvulkan` risolve.

---

## 3. Stato raggiunto (fine sessione)

- **Merge PR #557** (branch `pr-557-merge`, 8+ commit, autore originale Thiago
  Santos) — backend Vulkan C++: `vulkan/vulkan_backend.cpp`, singolo TU che
  include `_impl_gen.cpp` (stub generati) in coda.
- **`make vulkan` compila** tutti i binari (`ds4`, `ds4-server`, `ds4-bench`,
  `ds4-eval`, `ds4-agent`). Pipeline prefill+decode gira end-to-end su GPU con
  streaming pesi e LRU eviction.
- **Kernel test harness: 54/55 test PASS** (l'ultima riga di
  `vulkan/tests/results.txt` è `harness: 54 passed, 0 failed`; un test in
  passato falliva per un bug del test stesso — confronti `fabsf` su
  -inf/-nan — ora risolto).
- **Il modello produce il primo token reale non-BOS**:
  `<｜begin▁of▁sentence｜>庄园` (prima era solo BOS-loop).
  **MA la generazione si ferma dopo esattamente 1 token** (sia NTOK=8 che
  NTOK=30) — causa da diagnosticare (vedi sezione 9, punto 1).
- Pipeline: `prefill: 2.18 t/s, generation: 50.71 t/s` (riferimento quando
  tutto era stub ma funzionante; dopo i kernel reali la velocità è scesa a
  ~0.5-0.7 t/s perché molti kernel sono host-side su memoria mappata — la
  conversione a shader GPU è una fase FUTURA post-correttezza).

---

## 4. Architettura backend (`vulkan/`)

### 4.1 Contratto e stub

- Il motore richiede che TUTTI i simboli `ds4_gpu_*` dichiarati in
  `ds4_gpu.h` / `ds4_gpu_mgpu.h` siano definiti. Le funzioni non implementate
  nel backend sono generate come stub no-op (ritornano 1 senza scrivere gli
  output) da `python3 vulkan/gen_impl.py` → file `_impl_gen.cpp`.
- **Dopo aver aggiunto/rimosso una definizione `ds4_gpu_*` in
  `vulkan_backend.cpp`, rigenerare SEMPRE `python3 vulkan/gen_impl.py`**
  (altrimenti duplicate definition in link).
- Le definizioni nel backend devono stare **a colonna 0** (il generatore le
  rileva così; un leading space le fa mancare e genera lo stub duplicato).
- `kernel_status.py` (tabella) + `gen_impl.py` (stub) leggono il backend con
  regex.

### 4.2 Streaming pesi (modello di memoria)

- Heap GPU ~47 GiB vs modello ~81 GiB → streaming:
  `cache_model_range` registra solo metadata;
  `ensure_weight(offset, needed_bytes)` carica un range di pesi dalla mmap del
  modello in un buffer VMA al primo uso, con LRU eviction contro
  `DS4_VULKAN_WEIGHT_BUDGET_GB` (default 40).
- **`needed_bytes` DEVE essere la dimensione corretta del peso**:
  - f16 → 2 B/elemento
  - q8_0 → **34 B/blocco da 32 elementi** (f16 scale + 32×int8; NON 36)
  - q2_k → 84 B/blocco da 256
  - iq2_xxs → 66 B/blocco
  - f32 → 4 B/elemento
  Un errore corrompe il range del descriptor e il calcolo.
- **Clamp a `model_size - offset`**: mai leggere oltre la fine della mmap
  (SIGBUS).
- **Eviction**: non distruggere mai un buffer referenziato dal command buffer
  corrente (`g_vk.cmd_gen`, bump in `begin_cmd`; gli entry della generazione
  corrente sono saltati in eviction). RADV crasha al submit.

### 4.3 Semantica command buffer (contratto motore)

- Il motore avvolge il lavoro GPU come: `ds4_gpu_begin_commands()` → kernel →
  `ds4_gpu_end_commands()` (submit+wait) per token; alcuni path submit per
  layer.
- **A metà decode il motore chiama `ds4_gpu_commit_and_wait_selected_readback`
  e `ds4_gpu_flush_commands` e POI CONTINUA a registrare kernel senza un nuovo
  `begin_commands`** (semantica Metal: il prossimo encoder è implicito). Il
  backend quindi **ri-inizia un command buffer fresco dopo il submit** in
  queste due funzioni. Mai finalizzare due volte lo stesso CB.
- Un test kernel che dispaccia uno shader GPU DEVE avvolgere la chiamata in
  `begin_commands`/`end_commands`; altrimenti i comandi non vengono mai
  submit e l'output resta memoria stale.

### 4.4 Quirk RADV (mesa 26.0.2)

- `subgroupAdd` su `double` inaffidabile → riduzione workgroup plain (shared
  array + barrier + lane 0 sum).
- Range dei descriptor buffer devono essere esatti; accessi fuori range possono
  crashare RADV al command-stream finalize.
- `vkGetMemoryHostPointerPropertiesEXT` (VK_EXT_external_memory_host)
  **segfaulta su questo driver per qualunque puntatore** → niente external
  host memory.
- Weight upload: host → staging → `vkCmdCopyBuffer` → device (la destinazione
  è device-local anche su iGPU).

### 4.5 Shader

- GLSL in `vulkan/shaders/*.comp`, SPIR-V in `vulkan/shaders/spv/`.
  Compilazione: `glslangValidator -V --target-env vulkan1.2 <name>.comp -o
  spv/<name>.spv`. Il backend carica gli shader elencati in
  `load_all_shaders()` (nome + size push-constant); aggiungere lì i nuovi
  shader.
- I pesi f16 sono IEEE half impacchettati due-per-uint32; lo shader deve
  estrarre la metà bassa (elemento pari) o alta (elemento dispari).
- La GPU shader è usata solo per matmul_f16 e matmul_q8_0; gli altri kernel
  reali sono host-side su memoria mappata (come `add_tensor`) — accettati per
  la correttezza, la conversione a shader è fase futura.

### 4.6 Kernel tests

- Harness: `vulkan/tests/harness.cpp` + `vulkan/tests/tests.cpp`; ogni test è
  `vulkan/tests/tests/t_<nome>.cpp` con `REGISTER_TEST(name, fn)` e ritorna 0
  su PASS. Test sintetici → chiamano il kernel reale → confrontano con
  reference CPU inline.
- Run: `./run-kernel-tests.sh` (serve GPU, lo lancia l'utente); risultati in
  `vulkan/tests/results.txt`.
- Compile command (no GPU):
  ```
  g++ -O2 -g -std=c++17 -pthread -I. -Ivulkan -Ivulkan/include -DDS4_VULKAN_BUILD \
      vulkan/tests/harness.cpp vulkan/tests/tests.cpp vulkan/tests/tests/*.cpp \
      vulkan/vulkan_backend.cpp -lm -pthread -lvulkan -o vulkan-tests
  ```
- `ds4_gpu_add_tensor` è host-side (host add su memoria mappata), NON un
  dispatch GPU — non assumere "REAL" = GPU.

---

## 5. Bug importanti trovati e corretti (storico)

- **matmul_f16**: leggeva f16 come uint32 → fix (2 B/elemento) + riduzione
  workgroup (era la causa del BOS-loop insieme al punto sotto).
- **matmul_q8_0**: formato blocco 36→34 B + **rimozione del workaround
  `if (out_dim > 100000) return 1;`** (skippava l'output head vocab 129280 →
  era la causa principale del BOS-loop; workaround di debug di Thiago Santos
  rimasto in produzione).
- **rope_tail**: 4 bug (offset tail, frequenze, YaRN, attn_factor).
- **attention_output_q8_batch**: era un finto memcpy.
- **store_raw_kv_batch**: riga da bytes invece di head_dim + niente round-trip
  f16.
- **attention_prefill_raw_heads**: era un pass-through.
- **routed_moe_batch**: azzerava l'output → ora usa `ds4gk_routed_moe_slot`
  (condiviso col single-token).
- **hc_split_weighted_sum**: usava mixer grezzo invece di sinkhorn.
- **dsv4_fp8_kv_quantize**: era no-op → E4M3FN implementato.
- **Crash RADV `radv_amdgpu_cs_finalize`** su decode dopo grande prefill:
  il motore chiama commit/flush mid-decode e continua a registrare senza
  begin → il backend ri-inizia un CB fresco dopo il submit (sezione 4.3).
- **Bug nei test**: confronti `fabsf` su -inf/-nan fallivano (fix con
  `isinf`/`isnan`); allocazioni tensor troppo piccole nell'error-path di
  `routed_moe_batch`; ordine argomenti `ref_rope_tail` nel test fuso.

---

## 6. Kernel VERIFIED (31) — `VERIFIED` in `vulkan/kernel_status.py`

`ds4_gpu_matmul_f16_tensor`, `ds4_gpu_router_select_tensor`,
`ds4_gpu_routed_moe_one_tensor`, `ds4_gpu_attention_decode_heads_tensor`,
`ds4_gpu_embed_token_q8_0_tensor`, `ds4_gpu_embed_tokens_q8_0_tensor`,
`ds4_gpu_embed_token_quant_tensor`, `ds4_gpu_embed_tokens_quant_tensor`,
`ds4_gpu_indexer_score_one_tensor`,
`ds4_gpu_indexer_scores_decode_batch_tensor`, `ds4_gpu_indexer_topk_tensor`,
`ds4_gpu_attention_output_q8_batch_tensor`,
`ds4_gpu_attention_output_low_q8_tensor`,
`ds4_gpu_rms_norm_plain_rows_tensor`, `ds4_gpu_rms_norm_weight_rows_tensor`,
`ds4_gpu_rope_tail_tensor`, `ds4_gpu_head_rms_norm_tensor`,
`ds4_gpu_head_rms_norm_rope_tail_tensor`, `ds4_gpu_matmul_q8_0_tensor`,
`ds4_gpu_routed_moe_batch_tensor`, `ds4_gpu_dsv4_qkv_rms_norm_rows_tensor`,
`ds4_gpu_dsv4_fp8_kv_quantize_tensor`,
`ds4_gpu_store_raw_kv_batch_tensor`,
`ds4_gpu_attention_prefill_raw_heads_tensor`, `ds4_gpu_matmul_quant_tensor`,
`ds4_gpu_dsv4_qkv_rms_norm_rows_kv_rope_tensor`,
`ds4_gpu_kv_fp8_store_raw_tensor`, `ds4_gpu_matmul_q8_0_pair_tensor`,
`ds4_gpu_matmul_q8_0_hc_expand_tensor`,
`ds4_gpu_matmul_f16_pair_compressor_store_tensor`,
`ds4_gpu_matmul_f32_tensor`.

Rigenerare la tabella completa (con status VERIFIED/REAL/STUB per categoria)
con:
```
python3 vulkan/kernel_status.py > vulkan/STATUS.md.tmp
# oppure, come da workflow: troncare STATUS.md prima della tabella (## attention)
# e poi: python3 vulkan/kernel_status.py >> vulkan/STATUS.md
```
STATUS.md attuale riporta 261 funzioni, 102 real, 128 stub (dato leggermente
datato: include anche kernel REAL non ancora VERIFIED).

---

## 7. Script utili

| Script | Uso |
|---|---|
| `./run-test.sh` | rebuild + run modello. VAR: `MODEL`, `PROMPT`, `CTX`, `NTOK`, `DS4_VULKAN_WEIGHT_BUDGET_GB`, `SKIP_BUILD=1` |
| `./run-kernel-tests.sh` | build+run harness (scrive `vulkan/tests/results.txt`) |
| `python3 vulkan/gen_impl.py` | genera stub + set VERIFIED + tabella (rigenera `_impl_gen.cpp`, `_stubs.gen.cpp`) |
| `python3 vulkan/kernel_status.py` | tabella kernel per categoria |
| `vulkan/vk_mem_info.c`, `vulkan/vk_hostptr_test.c` | diagnostica memoria GPU |
| Debug env | `DS4_VULKAN_DEBUG=1` (trace begin/end/submit), `DS4_VULKAN_LOG_STUBS=1` (stub chiamati) |

---

## 8. Comandi di diagnosi

```sh
# harness (serve GPU — lo lancia l'utente)
./run-kernel-tests.sh

# smoke test modello
./run-test.sh                              # rebuild + 10 token
SKIP_BUILD=1 NTOK=4 ./run-test.sh          # veloce, no rebuild
DS4_VULKAN_WEIGHT_BUDGET_GB=46 ./run-test.sh
DS4_VULKAN_DEBUG=1 ./run-test.sh           # trace begin/end/submit
DS4_VULKAN_LOG_STUBS=1 ./run-test.sh       # stub chiamati
```

---

## 9. Prossimi passi

### 9.1 DIAGNOSI COMPLETATA (2026-08-01): fermo dopo 1 token = EOS da logits garbage

Sintomo: `<｜begin▁of▁sentence｜>庄园` poi stop (NTOK=8 e NTOK=30), nessun errore.

- **Causa (confermata dal codice)**: il loop in `ds4.c:46931`
  (`generate_metal_graph_raw_swa`) fa `sample_argmax(logits)` e, se il token è
  EOS (`<｜end▁of▁sentence｜>`), fa `break` **senza emettere nulla**
  (`vocab_token_is_generation_stop`, ds4.c:36656). Al 2° step i logits sono
  ancora garbage perché gli 8 stub non scrivevano gli output → argmax = EOS →
  stop silenzioso. NON è un crash/errore di sync.
- **Fix**: implementati gli 8 stub (sezione 9.2) — ora il decode deve produrre
  logits reali e non campionare più EOS.

### 9.2 FATTO (2026-08-01): implementati gli 8 stub del percorso critico

Tutti host-side su memoria mappata (come `add_tensor`), a colonna 0 in
`vulkan/vulkan_backend.cpp`, stub rimossi da `_impl_gen.cpp`, harness completo
compila (exit 0):

1. `ds4_gpu_matmul_q8_0_f16_out_tensor` — matmul Q8_0 (34 B/blocco) con output
   IEEE f16 (2 B/elem) via `ds4_float_to_half` (~riga 5053).
2. `ds4_gpu_matmul_f16_pair_tensor` — due matmul F16 paralleli f32 out
   (comp_kv + comp_sc), gestisce n_tok>1 (~riga 5101).
3. `ds4_gpu_hc_expand_split_half_tensor` — come hc_expand_split ma block_out_h
   f16; semantica split fast-path: `o = split[n_hc+h]*deq_f16(block_out) +
   residual` (~riga 2949).
4. `ds4_gpu_hc_expand_add_split_half_add_tensor` — add con block_add_h f16
   (~riga 2984).
5. `ds4_gpu_attention_output_q8_batch_f16_tensor` — come
   attention_output_q8_batch (stage A/B Q8_0) ma out_h f16 (~riga 2356).
6. `ds4_gpu_attention_prefill_static_mixed_heads_tensor` — prefill attention
   con raw_kv sliding window + comp_kv (f32/f16, `comp_kv_f16`), sink prior,
   softmax, layout MLA K==V (~riga 2463). Semantica comp: `raw_count =
   window ? min(window, t+1) : t+1`; `comp_count = min((t+1)/ratio, n_comp)`.
7. `ds4_gpu_dsv4_indexer_qat_tensor` — in-place per riga: Hadamard128
   (1/√128) + quantizzazione FP4 E2M1FN (blocchi da 32, scale = ldexp(1,
   ceil(log2(amax/6))), tie-to-even); head_dim DEVE essere 128 (~riga 3469).
8. `ds4_gpu_compressor_prefill_state_ratio4_tensor` — stato rolling ratio-4:
   azzera state_kv (8×width), state_score = -INF, copia le 4 righe tail in
   dst 0..3 (lane attention) con ape (f32/f16) sommato a state_score
   (~riga 5171). NB: dst 0..3 (NON 4..7, che è la semantica di store_batch).

Test nuovi (tutti in `vulkan/tests/tests/`):
`t_matmul_q8_0_f16_out.cpp` (matmul_q8_0_f16_out), `t_matmul_f16_pair.cpp`
(matmul_f16_pair), `t_hc_expand_split_half.cpp` (hc_expand_split_half),
`t_hc_expand_add_split_half_add.cpp` (hc_expand_add_split_half_add),
`t_attention_output_f16.cpp` (attention_output_q8_batch_f16),
`t_attention_prefill_static_mixed.cpp` (attention_prefill_static_mixed_heads,
5 casi: window/ratio piccoli, comp f16/f32, error-path),
`t_dsv4_indexer_qat.cpp` (dsv4_indexer_qat, _one_row, _bounds),
`t_compressor_prefill_state_ratio4.cpp` (compressor_prefill_state_ratio4,
_f16, _bounds).

PROSSIMO PASSO: l'utente lancia `./run-kernel-tests.sh` (serve GPU) e incolla
l'output; poi `SKIP_BUILD=1 NTOK=8 ./run-test.sh 2>&1 | tail -30` per vedere
se il modello ora genera più di 1 token.

### 9.3 Dopo la correttezza

- Convertire i kernel host-side in GPU shader per la velocità (fase futura,
  separata; oggi 0.5-0.7 t/s).
- Valutare commit del working tree su `pr-557-merge` (vedi sezione 11).

---

## 10. Workflow subagent (stabilito)

1. Un subagent per kernel (o batch paralleli di 3-4 su funzioni diverse dello
   stesso file — rischioso ma funziona).
2. Ogni subagent:
   - legge `AGENT.md` → sezione "Vulkan backend";
   - implementa il kernel host-side nel backend (NON esegue `gen_impl.py`, lo
     fa il coordinatore);
   - scrive il test `vulkan/tests/tests/t_<nome>.cpp` con
     `REGISTER_TEST(name, fn)` + reference CPU;
   - NON esegue GPU;
   - compila solo con il comando harness (sezione 4.6).
3. Il coordinatore: rigenera `gen_impl.py`, compila, l'utente esegue
   `./run-kernel-tests.sh`, su PASS si aggiunge il kernel a `VERIFIED` in
   `vulkan/kernel_status.py` e si rigenera `vulkan/STATUS.md` (troncare prima
   di `## attention` e ri-appendere con
   `python3 vulkan/kernel_status.py >> vulkan/STATUS.md`).

---

## 11. Stato git

- Branch: `pr-557-merge`. Commit recenti della sessione:
  `35f14b9 Finally a garbage output`, `53b9000 continue implementation`,
  `74be106 continue implementation`, `dcb1988 WIP vulkan`,
  `99503b0 remove vendored glslangValidator binary`,
  `d8ab59b Add updated skill archive with full debugging trace`,
  `d3f3820 Vulkan: implement llama.cpp-style CB rotation + pool cleanup...`,
  `2c2ef0c FIX: vkResetDescriptorPool between prefill/decode...`,
  `1761a18 Clean matmul_f16 dispatch...`, `9e415b7 FIX: use separate
  descriptor set for matmul_f16...`, `2b41c15 Vulkan: add simple decode-only
  matmul shader...`, `22bdffc Vulkan: sync shaders+C++ 2D grid...`,
  `9c87ee5 feat: implementing vulkan backend`.
- Ultimo `git status` (fine sessione): modifiche residue non tracciate
  `vulkan-tests` (binario), e untracked `.envrc`, `download-laguna.sh`,
  `guix.scm`. La maggior parte del lavoro è già committata.
- Nota: `misc/COMPACT.md` documenta la politica di compattazione contesto di
  ECA (non è un file di stato del backend).

---

## 12. Comunicazione con l'utente

- Rispondere in italiano.
- La macchina di test è la shell utente con `/dev/dri`; gli agenti non possono
  eseguire GPU: chiedere all'utente di lanciare i comandi e incollare
  l'output.
- Preferire `g++`/`gcc` (niente `cc`).
