---
status: current
code: [src/app/board_batch.cpp, src/app/batch_main.cpp, src/gpu/fixed]
reviewed: 2026-11-08
---

# Route independent boards in parallel

`tracemaker-batch` routes several boards in one process. Each board has one routing thread,
one variant, its own rule context and independent output files. The selected GPU serves their
fields and supported fixed-geometry queries concurrently. A* and route decisions run on the CPU.

Build the normal application targets with CUDA enabled, then select the GPU by UUID:

```sh
cmake --build build/release --target tracemaker tracemaker-batch --parallel 2
CUDA_VISIBLE_DEVICES=GPU-your-device-uuid build/release/src/app/tracemaker-batch \
  jobs.json --workers 32 --device 0 --report results.json
```

`--device 0` means the first device visible to this process. Selecting one UUID makes all board
workers share that physical GPU. `--workers` accepts 1–256 and never starts more workers than boards.
This is an interface limit, not a guarantee that 256 boards fit in device or host memory.

## Prepare the manifest

Use a nonempty JSON array. `work` must be positive; `seed` defaults to 1.
Paths are relative to the working directory, so absolute paths are useful for unattended runs.
Keep each input's `.kicad_pro` and `.kicad_dru` beside its `.kicad_pcb`; they supply its original rules.

```json
[
  {
    "id": "board-a",
    "input": "inputs/a/unrouted.kicad_pcb",
    "output": "outputs/a/routed.kicad_pcb",
    "work": 100000000,
    "seed": 1
  },
  {
    "id": "board-b",
    "input": "inputs/b/unrouted.kicad_pcb",
    "output": "outputs/b/routed.kicad_pcb",
    "work": 100000000,
    "seed": 1
  }
]
```

The entire manifest is validated before creating output directories. IDs must be unique, inputs must
exist, and output families must neither overlap one another nor overwrite inputs or existing files.
An output family includes its board, project, rules, `.kicad_pcb.json` summary and `.kicad_pcb.log`.
The combined report and its `.progress.jsonl` journal must also have unused paths.

## Follow the run

Each completed board appends and flushes a line to `results.json.progress.jsonl`.
The final report preserves manifest order and includes each board's summary and routing interval,
the actual worker count, peak simultaneous routes, elapsed time and throughput.
A per-board exception is recorded; other boards continue. Exit codes are 0 for complete routing,
3 when processing completed with open connections, and 1 for processing or setup errors.
`complete` in the combined report means every board was processed successfully, not that all nets connected.

Rescue passes and open-connection explanations retain the single-board command's defaults.
For comparisons at the requested fixed work budget, pass both `--no-rescue --no-why-open`.
Each board has a 3600-second safety stop; reject time-limited runs when comparing deterministic work.
There is no knowledge base in the batch interface.

## Check identical results

Use two manifests with the same inputs, work and seeds, but different output directories.
First run `--workers 1`, then the desired concurrency. `--no-gpu` selects CPU reference fields
and disables CUDA fixed-query prefetch. Diagnostic `--verify-acceleration` additionally checks GPU
field values, consumed fixed-query answers and conservative routed-geometry predictions against CPU references;
its extra work makes it unsuitable for timing runs.

```sh
python3 tests/routing/compare_routes.py serial/a/routed.kicad_pcb parallel/a/routed.kicad_pcb \
  --reference-summary serial/a/routed.kicad_pcb.json \
  --candidate-summary parallel/a/routed.kicad_pcb.json
```

The comparator checks complete board/project/rules bytes and deterministic route/work summaries.
Compare every board in the manifest. GPU arrival order may change prefetch hit counts and timing;
it must not change the routed files, work budget or routing decisions.
See the [measured results](../../bench/records/routing-acceleration/README.md) and
[implementation details](../07-gpu.md#9-exact-cuda-routing-and-independent-board-batches).
