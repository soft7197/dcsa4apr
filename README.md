# Agent-Based Automated Program Repair with Dynamic Context and Static Analysis


This repository contains the source code and experimental results for our agent-based automated program repair framework. The system orchestrates three specialized LLM agents — **Context Updater (CU)**, **Generator (G)**, and **Overfitting Detector (OD)** — with six static analysis tools and a dynamic context pool to iteratively repair software bugs.

### Key Results (Defects4J v1.2 + v2.0, 660 bugs)

| Model | Defects4J v1.2 | Defects4J v2.0 | Total |
|-------|---------------|---------------|-------|
| **GPT-4o** | 187 | 178 | **365** |
| **QwenCoder-32B** | 131.3 ± 3.1 | 120.7 ± 2.1 | **252** |
| **CodeLlama-34B** | 60.7 ± 3.5 | 47.3 ± 4.0 | **108** |

(Open-source model results: mean ± std across 3 independent runs)

---

## Architecture

<p align="center">
  <img src="docs/Architecture.png" alt="Overall Architecture" width="85%"/>
</p>

**SA**: Static Analysis &nbsp;|&nbsp; **CU**: Context Updater &nbsp;|&nbsp; **G**: Generator &nbsp;|&nbsp; **OD**: Overfitting Detector

- **Iteration 1**: G operates on static context only (buggy code + tests + errors).
- **Iteration 2+**: CU analyzes failures → selects tools → retrieves context → G creates patches.
- **On test pass**: OD validates semantic correctness → if overfitting detected, triggers G refinement.

---

## Agents and Prompt Templates

Complete prompt templates for all three agents are documented in [`docs/PROMPT_TEMPLATES.md`](docs/PROMPT_TEMPLATES.md).

## Model Configurations

### GPT-4o (Closed-Source)

| Parameter | Context Updater | Generator | Overfitting Detector |
|-----------|----------------|-----------|---------------------|
| **Model** | `gpt-4o` | `gpt-4o` | `gpt-4o` |
| **Temperature** | 0.2 | 1.0 | 0.1 |
| **n** (completions) | 1 | 10 | 1 |
| **response_format** | `json_object` | `json_object` | `json_object` |
| **API** | OpenAI API | OpenAI API | OpenAI API |

### QwenCoder-32B (Open-Source)

| Parameter | Context Updater | Generator | Overfitting Detector |
|-----------|----------------|-----------|---------------------|
| **Model** | `Qwen/QwenCoder-32B-Instruct` | `Qwen/QwenCoder-32B-Instruct` | `Qwen/QwenCoder-32B-Instruct` |
| **Temperature** | 0.2 | 0.7 | 0.1 |
| **n** (completions) | 1 | 10 | 1 |
| **response_format** | `json_schema` (via vLLM proxy) | `json_schema` (via vLLM proxy) | `json_schema` (via vLLM proxy) |
| **max_tokens** | — | 800 (single) / 1500 (multi) | 600 |
| **Deployment** | vLLM on 2× A6000 GPUs | vLLM on 2× A6000 GPUs | vLLM on 2× A6000 GPUs |

### CodeLlama-34B-Instruct (Open-Source)

| Parameter | Context Updater | Generator | Overfitting Detector |
|-----------|----------------|-----------|---------------------|
| **Model** | `codellama/CodeLlama-34b-Instruct-hf` | `codellama/CodeLlama-34b-Instruct-hf` | `codellama/CodeLlama-34b-Instruct-hf` |
| **Temperature** | 0.2 | 0.7 | 0.1 |
| **n** (completions) | 1 | 10 | 1 |
| **response_format** | `json_schema` (via vLLM proxy) | `json_schema` (via vLLM proxy) | `json_schema` (via vLLM proxy) |
| **max_tokens** | — | 800 (single) / 1500 (multi) | 600 |
| **Deployment** | vLLM on 2× A6000 GPUs | vLLM on 2× A6000 GPUs | vLLM on 2× A6000 GPUs |

### Open-Source Model Serving Architecture

Open-source models are served via a local proxy (`server.py`) with the following architecture:

```
Client (port 8000) → FastAPI Proxy → vLLM (port 8001)
```

The proxy provides:
- **JSON schema enforcement**: Forces structured output matching patch response schemas
- **Response coercion**: Post-processes type mismatches and malformed JSON
- **Timeout retry**: Retries slow vLLM calls (max 3 retries, 30-min timeout)

**vLLM Server Parameters:**
```
--tensor-parallel-size 2      # 2× A6000 GPUs
--dtype bfloat16
--gpu-memory-utilization 0.95
--max-model-len 16384
--enable-prefix-caching        # Cache system prompt KV across requests
--enable-chunked-prefill       # Better GPU utilization for mixed prompt lengths
```

### Shared Hyperparameters (All Models)

| Parameter | Value |
|-----------|-------|
| Max iterations per bug | 5 |
| Candidates per iteration (n) | 10 |
| Max hypothesis pool size | 10 |
| Knowledge base token limit | 10,000 |
| Max refinement attempts | 1 |
| Patch deduplication | By normalized whitespace |
| Early stopping | Enabled |

---

## Experimental Results

Patch result JSONs are included in `results_gpt4o/` and `results_open_source/`. See [`docs/RESULTS.md`](docs/RESULTS.md) for detailed analysis.

### Correct Patch Evaluation

Human-readable evaluation of all semantically correct patches is provided in [`correct_patches/`](correct_patches/), organized by model and experiment. Each entry shows the buggy method, correct patch, and a short semantic equivalence decision against the ground-truth developer patch.

> **Note:** Execution traces (~1.5 GB), vector databases, and Defects4J checkouts are excluded from this repository due to size. They are generated automatically during preprocessing and repair.


---

## Acknowledgments

This work was supported by the National Research Foundation of Korea (NRF) grant funded by the Korea government (MSIT) (NO.2020R1A2B5B01002467 and NO. RS-2022-NR068754).

