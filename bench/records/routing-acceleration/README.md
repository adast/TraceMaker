# Exact routing acceleration

This record separates CPU algorithm improvements, single-board CUDA acceleration and throughput across
32 independent boards. Timings, complete artifact hashes, deterministic work and native route-oracle
comparisons are recorded together. Benchmark boards are not redistributed.

## Implemented optimizations

| Optimization | Executes on | Change and invariant |
|---|---|---|
| Outline row intervals and prefix coverage | CPU | Classifies the original cell centres from the same outline/cutout crossings; preserves boundary rules |
| Radix priority partitions with a tie heap | CPU | Preserves signed priority and unsigned state-key ordering, including nonmonotone inserts and field re-keying |
| Packed conservative stale snapshots | CPU | Proves some stale entries early without enlarging the 16-byte queue record; retains original tests and overflow-domain fallback |
| Complete cell-cost memoization | CPU | Reuses the full cost within the original generation and scratch storage |
| Planar move masks and cost tables | CPU | Precomputes valid transitions and representable integer costs without changing neighbor order |
| Zero-backed search-node storage | CPU | Initializes the original trivial node records through zero-backed allocation; no parent compression |
| Conservative routed-copper and hole bounds | CPU | Skips exact traversal only when no foreign geometry can intersect; uncertain/rule-dependent cases use the scalar path |
| Warp prefix-min distance fields | GPU | Processes 32 positions at a time, carries segmented minima between tiles and pairs opposite sweeps |
| Device-side field convergence | GPU | Keeps ordered sweeps, via relaxation and convergence in a resident cooperative solve; ordinary-kernel and CPU fallbacks remain |
| Fixed polygon/edge indices | CPU preparation and GPU query | Reduces exact immutable geometry queries to relevant polygon edges; retains original integer predicates |
| Immutable index reuse | CPU | Reuses pure geometry indices only after full value equality |
| Resident fixed scenes and asynchronous point tiles | CPU and GPU | Each router owns its resources and consumes only ready answers; misses and unsupported contexts use CPU predicates |
| Independent-board scheduling | CPU hosts and one GPU | Isolates routers, rules, mutable state and outputs; direct CUDA calls allow concurrent independent work |

Contributions are measured for the combined implementations. The rows do not claim isolated speedups,
and their effects must not be multiplied. A* queue operations, search decisions, mutable geometry,
rip-up, exact commit checks and final board writing remain CPU work.

## Alternatives excluded from the production branch

These summarize the completed prototype study; they are not measurements of this implementation.

| Alternative | Executes on | Why excluded |
|---|---|---|
| GPU A* and bulk GPU search-state initialization | GPU and CPU | Exact ordering, model preparation, visibility and refinement overhead prevented a useful whole-router win |
| Shared cross-board field service | CPU dispatch and GPU | Direct per-board calls gave the accepted throughput with simpler ownership |
| Brute-force GPU geometry | GPU | Visiting every polygon edge erased the benefit; the accepted evaluator uses spatial indices |
| GPU via queries and routed-via offload | CPU and GPU | Preparation, materialization and fallback costs prevented a stable accepted end-to-end gain |
| GPU rasterization | GPU | Retained only as a prototype comparator; exact CPU row intervals were selected |
| More compact parent/state layouts | CPU | Capacity reductions did not establish a stable routing-time gain; original node representation retained |
| Alternative queue minimum scans and extra caches | CPU | No stable whole-routing gain over the selected simple minimum scans |
| Field transpose, convergence-flag and readback experiments | GPU and host API | Component signals did not establish a stable accepted complete-routing gain |
| Compiler/profile optimization machinery | Build | Excluded from this branch; both measured builds use the same compiler policy |

## Measurement protocol

The reference is unmodified fork `main` at `057ee8208afe76c26838ee08934906938b17c613`.
The measured implementation is the ancestor identified in `results.json`; the final documentation
commit does not change or relabel that implementation.

Both builds use the same GCC release configuration, `-O3`, and `-ffp-contract=off`, with no profile
feedback or compiler tuning. The GCC 14 reference emits `nonnull` and `maybe-uninitialized` diagnostics;
only those diagnostics are nonfatal for both remote builds. The per-commit GCC 15 builds retain warnings
as errors. CUDA uses CUDA 12.4 and the same host compiler for both builds.
All native runs use the same machine and one selected A100. CPU-only trials are serial.
Single-board and sequential runs use one logical CPU; the 32-board runs use 32 logical CPUs.
The host has two Xeon Platinum 8380 processors and an A100-SXM4-80GB; `hardware.json` records the device and driver.

The ten-board comparison includes four modes: original CPU, original GPU, accelerated CPU and accelerated
GPU. It retains each board's original project/rules, seed and full work budget.
It uses one route thread and variant, no knowledge base, `--no-rescue --no-why-open`, and a
3600-second safety stop. One timing trial per mode per board is reported; these are validation
measurements, not confidence intervals. Original GPU controls run in a separate serialized pass after
the primary measurements. Each CPU/GPU result must match the reference CPU's
complete PCB/project/rules hashes, deterministic summary/work and native legality/reachability diagnostics.

The 32-board comparison uses the same manifest and candidate binary: an actual sequential run,
then two runs with 32 workers. Each board retains its work and context. Every output is compared
with its sequential counterpart. Native counters are compared as complete process-wide multisets.
Peak concurrency and the common interval where all 32 route
calls overlap are checked; only one physical GPU is visible. This factor includes 32 CPU host
threads and GPU work; it is throughput, not GPU A* acceleration.

An additional untimed full-budget route checks accelerated field values, consumed fixed-query
answers and routed-bound predictions against the CPU references. Positive verification counters
prove activation. Diagnostic runs are excluded from performance tables.

## Results

All comparisons below use complete process wall time, including startup and board I/O.
The single-board values are one trial per mode and board; summary factors are medians of within-board ratios.
The CPU-only and GPU tables share the same inputs and full work budgets.

### CPU-only improvements

Median speedup versus original CPU: **1.314×**, range 1.221–1.370×.

| Board | Original CPU, s | Our CPU, s | Speedup |
|---|--:|--:|--:|
| Microdox-PCB_Microdox | 127.959 | 98.291 | 1.302× |
| LeeChee_1800 | 238.051 | 195.018 | 1.221× |
| EUC-VESC_electronics_sept | 198.565 | 161.977 | 1.226× |
| xmasOrn_xmasOrn | 153.564 | 114.029 | 1.347× |
| Own-Mailbox-Hardware_pierre | 173.591 | 126.676 | 1.370× |
| kitspace_neotron-32 | 169.684 | 137.859 | 1.231× |
| youtubeplaybutton_playbutton | 192.183 | 144.843 | 1.327× |
| VB-IoT1_VB-IoT1_v1 | 154.351 | 121.129 | 1.274× |
| 8088_sbc2_pcb_8088_sbc2 | 179.003 | 133.617 | 1.340× |
| ATMEGA328-Motor-Board_ATMEGA328_Motor_Board | 196.965 | 148.209 | 1.329× |

### Single-board GPU improvements

Median original-GPU/our-GPU factor: **1.772×**.
The complete accelerated GPU path is **1.695×** faster than original CPU,
and **1.293×** faster than our CPU-only mode, by the same within-board statistic.
These factors have different denominators and must not be multiplied.

Both GPU modes retain CPU A*. Original GPU uses the unchanged fork defaults, including the
500,000-cell CUDA field cutoff; our optimized CUDA solver uses its accepted zero cutoff.
The resulting field values, routing decisions and charged work match.

| Board | Original GPU, s | Our GPU, s | Speedup |
|---|--:|--:|--:|
| Microdox-PCB_Microdox | 135.505 | 78.043 | 1.736× |
| LeeChee_1800 | 203.959 | 104.310 | 1.955× |
| EUC-VESC_electronics_sept | 199.636 | 101.237 | 1.972× |
| xmasOrn_xmasOrn | 169.372 | 101.032 | 1.676× |
| Own-Mailbox-Hardware_pierre | 191.425 | 96.269 | 1.988× |
| kitspace_neotron-32 | 174.963 | 99.521 | 1.758× |
| youtubeplaybutton_playbutton | 203.620 | 114.068 | 1.785× |
| VB-IoT1_VB-IoT1_v1 | 167.882 | 83.148 | 2.019× |
| 8088_sbc2_pcb_8088_sbc2 | 175.192 | 119.066 | 1.471× |
| ATMEGA328-Motor-Board_ATMEGA328_Motor_Board | 197.683 | 121.847 | 1.622× |

### 32-board GPU throughput

Actual sequential processing: **1981.761 s**. Median concurrent time: **126.057 s**;
throughput factor **15.721×** against that measured sequential run.
The manifest, candidate binary, board work and GPU are identical across these runs.

| Run | Concurrent boards | Wall time, s | All-32 overlap, s | Exact versus sequential |
|---|--:|--:|--:|---|
| Sequential | 1 | 1981.761 | — | reference |
| wide-concurrent-a | 32 | 123.393 | 67.769 | 32/32 |
| wide-concurrent-b | 32 | 128.721 | 68.011 | 32/32 |

### Validation and scope

- All ten single-board cases match across original CPU/GPU and our CPU/GPU: complete PCB/project/rules bytes, deterministic work and native counters.
- All 64 concurrent batch outputs match their actual sequential counterparts, with complete process-wide native-counter multisets preserved.
- The 30-board quick-set regression matches in original CPU, our CPU and our GPU modes at 1M work, eight variants and one thread.
- Every implementation commit passes its full available test suite with zero local compiler warnings. The final source passes 420 tests, skips 43, and passes all 13 sanitizer checks.
- Compiler warnings: original 2; ours 1 (all inherited); **new 0**.
- Full-budget scalar verification checks 286,579,382 GPU field values, 2,303,030 consumed fixed-geometry values and 4,623,055 conservative routed-bound predictions.

These runs verify equivalence to the fixed reference; they do not provide a fresh KiCad DRC judgment.
Metal and 256-board performance were not tested. Shared-machine load and one single-board timing per
mode limit timing precision; original GPU controls follow the primary measurements in a separate pass.
No per-algorithm contribution or isolated kernel ratio is inferred from the combined routing times.

## Reproduction and evidence

Use the [batch recipe](../../../docs/how-to/board-batches.md) and `tests/routing/compare_routes.py`.
`inputs.json` gives exact input/context hashes, seeds and work; `build.json` identifies the measured
implementation/reference and executable hashes. `exactness.json` retains every primary case signature,
artifact hash and full native process multiset. `reference-gpu.json` and `quick-equivalence.json` retain
the additional comparisons. `validation.json`, `component-checks.json` and `compiler-warnings.json`
record the test and warning gates. Raw routed boards and logs remain outside Git.
Exact replay requires matching input boards and their project/rules files.
