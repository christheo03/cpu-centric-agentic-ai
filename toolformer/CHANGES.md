# Toolformer benchmark: changes after run 22385

## Why we changed things

Run 22385 (2× V100, full sweep BS 1→128) finished, but the numbers did not match the paper
("Towards Understanding, Analyzing, and Optimizing Agentic AI Execution: A CPU-Centric Perspective"):

- BS=1 end-to-end latency was **47 s** (paper: a few seconds).
- vLLM generated only **~10 tok/s** for a single request, and only 11–50 tok/s in total with 128 requests.
- Throughput peaked at **BS=32** (0.061 req/s) and dropped at BS=64/128. In the paper, Toolformer throughput keeps rising up to BS=128.
- vLLM log: `Cannot use FA version 2 ... compute capability >= 8` → `Using FlexAttention backend`. The V100 (compute capability 7.0) cannot use FlashAttention-2, so vLLM fell back to FlexAttention, which is the likely cause of the slow generation.
- Every LLM call generated **256 tokens** (LangChain default), although the code only uses the first line.

---

## 1. Multi-processing → multi-threading (`batch_toolformer.py`)

**Why:** the paper (Section 3.3.2) uses multi-threading (MT) for Toolformer, not multi-processing (MP):
> "MT approach works better for I/O workloads. Therefore, we select MT for Toolformer as it contains an I/O tool, i.e. the WolframAlpha API."

Each request in a batch spends its time waiting on HTTP (vLLM and Wolfram). The Python GIL is released while
waiting, so threads run in parallel without the cost of starting one process per request (128 processes on 24 cores at BS=128).

This parallelises the **whole request** (LLM → Wolfram → LLM), one thread per problem in the batch.
The `ThreadPoolExecutor` inside `execute_calculations` in `math_toolformer.py` (line 267) was already
multithreaded. It only runs several calculator calls of the same problem in parallel, and it was not changed.

Line 18, import:
```python
# before
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError
# after
from concurrent.futures import ProcessPoolExecutor, as_completed, TimeoutError, ThreadPoolExecutor
```

Line 206, executor:
```python
# before
with ProcessPoolExecutor(max_workers=batch_size) as executor:
# after
with ThreadPoolExecutor(max_workers=batch_size) as executor:
```

(The comment on line 204–205 still says "ProcessPoolExecutor / own process" and is out of date.)

---

## 2. Generation settings (`math_toolformer.py`, lines 206–212)

```python
# before
self.llm = VLLMOpenAI(
    base_url=base_url,
    model=model_path,
    openai_api_key='EMPTY'
)
# after
self.llm = VLLMOpenAI(
    base_url=base_url,
    model=model_path,
    openai_api_key='EMPTY',
    temperature=0,
    max_tokens=100
)
```

- **`max_tokens=100`** (was the LangChain default of 256, from `BaseOpenAI` in
  `langchain_community/llms/openai.py`). GPT-J rarely stops on its own, so this caps wasted generation.
- **`temperature=0`** (was the default of 0.7). The output is now deterministic, so runs can be repeated and compared.
- **No `stop` sequence, by choice.** Most calls will generate the full 100 tokens, so the LLM cost per call is
  fixed and consistent. This inflates the LLM share of latency compared with stopping at the first line, so state it when comparing with the paper.

---

## 3. Single GPU instead of two (`toolformer.sh`)

```bash
# before
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:2
    --tensor-parallel-size 2 \
# after
#SBATCH --gres=gpu:tesla_v100-sxm2-32gb:1
    --tensor-parallel-size 1 \
```

**Why:**
- The paper's systems each use one GPU (RTX Pro 6000 or H200).
- GPT-J-6B in fp16 is ~12 GB and fits on one 32 GB V100.
- Tensor parallelism makes the two GPUs exchange data at every layer for every token, which is costly for a small model.
- One free V100 is easier to get in the queue than two.

**Trade-off:** KV cache drops from ~104k tokens to roughly ~30k. At BS=128 some requests may wait for cache space, which is a real GPU-memory saturation effect (paper Section 3.3.1).

---

## 4. Attention backend (`toolformer.sh`, line 41)

```bash
# added
export VLLM_ATTENTION_BACKEND=TRITON_ATTN
```

**Why:** forces vLLM to use its Triton attention kernels instead of falling back to FlexAttention.
This is an experiment. If `TRITON_ATTN` is not supported on V100 in vLLM 0.11, vLLM will fail to start (see change 5).
Fallbacks: remove the line; or use vLLM 0.10.x with `VLLM_USE_V1=0`.

---

## 5. Stop the job if vLLM crashes (`toolformer.sh`, lines 60–65)

```bash
# before
until curl -s http://127.0.0.1:5000/v1/models > /dev/null; do
    sleep 10
done
# after
until curl -s http://127.0.0.1:5000/v1/models > /dev/null; do
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM crashed during startup - check the log above"; exit 1
    fi
    sleep 10
done
```

**Why:** before, if vLLM crashed at startup the loop waited until Slurm's time limit and the whole job allocation was wasted.

---

## 6. Short test run first (`toolformer.sh`)

```bash
# before
#SBATCH --time=05:00:00
python batch_toolformer.py --output ./benchmark_results_4b.json --timeout 2800
# after
#SBATCH --time=01:30:00
python batch_toolformer.py --output ./benchmark_results_test.json --batch-sizes 1 8 32 --timeout 2800
```

**Why:**
- Checks that the fixes work before spending hours on the full sweep.
- Writes to a new file, so the results of run 22385 (`benchmark_results_4b.json`) are kept for comparison.
- A shorter time limit lets Slurm start the job early in gaps between other jobs (backfill).

Also commented out "TEST 1" (`math_toolformer.py` single-batch run), which was already disabled.

---

## Earlier change (made before this review)

`math_toolformer.py`, lines 742 and 782: calculator latency printed with 6 decimals (`:.6f`) instead of 3 (`:.3f`).

---

## Jobs

| Job | Setup | Result |
|---|---|---|
| 22385 | 2× V100, ProcessPool, 256 tokens, FlexAttention, BS 1→128 | Finished; too slow, see top |
| 24878 | 2× V100, new settings, 5 h | Cancelled (switched to 1 GPU) |
| 24879 | 1× V100, 5 h | Cancelled (switched to 1.5 h) |
| 24880 | 1× V100, 1.5 h, BS 1/8/32 | Submitted, waiting for a GPU |

## What to check in job 24880's log

1. vLLM started (no "vLLM crashed during startup" line).
2. The attention backend in the startup log is not FlexAttention.
3. BS=1 end-to-end latency is a few seconds, not 47 s.
4. `initial_solution` in `benchmark_results_test.json` still contains `[Calculator(...)]`.

If all four are fine: set `--time` to ~03:00:00, run all batch sizes (1→128) with a new output name.

## Settings to report with the results

GPT-J-6B, fp16, vLLM 0.11.0, 1× V100-32GB, 24 CPU cores, `TRITON_ATTN` backend (if it works),
temperature 0, max_tokens 100, no stop sequence, multi-threading (one thread per request),
MAWPS dataset, WolframAlpha API calculator.
