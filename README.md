# CacheSolidarity: Preventing Prefix Caching Side Channels in Multi-tenant LLM Serving Systems

Artifact repository for the paper:

> **CacheSolidarity: Preventing Prefix Caching Side Channels in Multi-tenant LLM Serving Systems**
> Panagiotis Georgios Pennas (IMDEA Software Institute, Universidad Politécnica de Madrid),
> Konstantinos Papaioannou (Universidad Politécnica de Madrid & IMDEA Software Institute),
> Marco Guarnieri (IMDEA Software Institute),
> Thaleia Dimitra Doudali (IMDEA Software Institute)
>
> *Proceedings of the ACM International Middleware 2026 Conference.*

This repository contains the source code, scripts, and supporting materials required to reproduce the experimental evaluation presented in the paper. A ready-to-run Docker image is available at [`panapenn/cachesolidarity`](https://hub.docker.com/r/panapenn/cachesolidarity). The artifact was created specifically to support the paper's evaluation and is intended to facilitate artifact evaluation, reproducibility, and reuse.

---

## 📌 Overview

This artifact provides a complete, containerized framework to reproduce CacheSolidarity's security and performance evaluation. It is designed so that the experiments can be executed end-to-end via a single Docker image, without requiring manual modification of the source code.

In particular, the artifact provides:

- **The CacheSolidarity implementation on top of vLLM** (0.8.5, V1 engine): the KV Cache Extension, Detector, and Activator.
- **Re-implementations of the evaluated baselines**: Automatic Prefix Caching (no security), User Cache Isolation, and SafeKV (semantic-aware selective sharing).
- **Scripts for building the privacy-evaluation datasets** and running the PII-masking/detection pipeline used to measure security.
- **Scripts for executing each performance-evaluation experiment** (cross-workload, cross-model, request-rate sensitivity, threshold sensitivity).
- **A script to automate the execution of the full evaluation** (`run.sh`).
- **A Jupyter notebook** (`plots.ipynb`) that parses the results and generates every figure/table reported in the paper.

---

## 🏆 Artifact Evaluation

This artifact is submitted for evaluation under the following criteria:

- **Artifacts Available**
- **Artifacts Functional**
- **Results Reproduced**

> **For artifact evaluators:** The recommended evaluation path is described in [Getting Started](#-getting-started).

### Relationship between the artifact and the paper

| Paper component | Artifact component | Reproduction instructions |
|---|---|---|
| Table 1 | `src/pii_masking.py`, `src/bert_pii_rank.py` | [Experiment 1](#claim-1--security-of-selective-isolation) |
| Figure 6 | `src/safeKV_privacy_eval.py` | [Experiment 1](#claim-1--security-of-selective-isolation) |
| Figure 3 | `run.sh` (Figure-3 block) | [Experiment 2](#claim-2--conditional-exploitability-of-apc-timing-leaks) |
| Figure 7 | `run.sh` (Fig-7 block) | [Experiment 3](#claim-3--performance-preservation-across-workloads) |
| Figure 8 | `run.sh` (Fig-8 block), `run_70b.sh` | [Experiment 4](#claim-4--performance-preservation-across-models) |
| Figure 9 | `run.sh` (Fig-9 block) | [Experiment 5](#claim-5--robustness-to-request-rate) |
| Figure 10 | `run.sh` (Fig-10 block) | [Experiment 6](#claim-6--securityperformance-trade-off-via-the-activator) |

> **Disclaimer:** Full-scale results across all evaluated LLM models (up to Llama3-70B on 4×H100 GPUs) require substantial multi-GPU resources and multi-day runtimes. Reviewers without access to comparable hardware are encouraged to reproduce the artifact on a subset of models/workloads (e.g., the smaller models and `workload_avg`), which demonstrates the same qualitative trends reported in the paper.

---

## 📂 Repository Structure

```
.
├── run.sh               # Full evaluation (single GPU)
├── run_70b.sh           # Llama3-70B experiments (multi-GPU, TP = 2)
├── plots.ipynb          # Figure/table generation
├── requirements.txt
├── requirements_2.txt
├── data/                # Workload traces
├── src/                 # Security pipeline + performance driver
├── llmperf/             # Benchmarking/output utilities
└── vllm/                # vLLM 0.8.5 with CacheSolidarity and baselines
```

### Top-level files

| File | Description |
|---|---|
| `run.sh` | Master script orchestrating the full evaluation (Phase 1: security pipeline, Phase 2: performance sweeps). |
| `run_70b.sh` | Llama3-70B experiments for Figures 3 and 8 (requires 2+ GPUs, tensor parallelism TP = 2). |
| `plots.ipynb` | Parses the results under `temp_results/` and produces the final figures/tables to directly compare with the paper. |
| `requirements.txt`, `requirements_2.txt` | Python dependencies (already installed in the Docker image). |
| `models/` | Mount point for downloaded model weights (not included; see [Getting Started](#-getting-started)). |

### `vllm/`

The modified vLLM 0.8.5 (V1 engine) containing the CacheSolidarity implementation (KV Cache Extension, Detector, Activator) and the APC, User Cache Isolation, and SafeKV baselines. Per-user bookkeeping lives in `vllm/user_metadata.py`, `vllm/user_monitor.py`, and `vllm/request_metadata.py`. Precompiled CUDA kernels (`*.so`) are not stored in this repository; they are included in the Docker image.

### `src/`

Contains the implementation of the security-evaluation pipeline and the performance-evaluation driver.

| File | Description |
|---|---|
| `build_datasets.py` | Builds the privacy-evaluation datasets (Flan, UltraChat, ShareGPT, OASST1) into `data/privacy_evaluation/`. |
| `patch_ai4privacy.py` | Patches the ai4privacy PII-detection library used to identify secrets in prompts. |
| `pii_masking.py` | Masks and replaces sensitive words in each dataset (Table 1). |
| `bert_pii_rank.py` | Matches masked entries against candidate reconstructions (Table 1). |
| `safeKV_privacy_eval.py` | Runs the PII-detection/recovery evaluation behind Figure 6, comparing CacheSolidarity against SafeKV. |
| `performance_evaluation_with_timestamps.py` | Main experiment driver; run per workload/model/request-rate combination with `--sys_mode` set to `APC`, `CacheSolidarity`, `Isolation`, or `SafeKV`. |
| `download_models.py` | Helper that downloads all evaluated models from the Hugging Face Hub. |

### `data/`

| Directory | Description |
|---|---|
| `exploitability/` | Prompt trace for the exploitability study (Figure 3). |
| `workloads/` | Multi-tenant traces `workload_1`–`workload_5` and `workload_avg` (Figures 7–9). |
| `sensitivity/` | Trace for the Activator threshold sensitivity analysis (Figure 10). |

### `temp_results/`

Created at runtime. Raw results are written here, one subdirectory per experiment (e.g. `fig7_workloads/`, `fig8_models/`, `fig9_RPS/`, `fig10_theta_performance/`, `Figure_3_exploitability/`, `pii_results/`).

### `plots/`

Created by `plots.ipynb`. Contains the figures generated from the experimental results. These are expected to match the trends of Figures 3, 6–10 and Table 1 of the paper.

---

## 💻 Hardware Requirements

This artifact requires an NVIDIA GPU and has been tested on the following configuration:

- **Single-GPU experiments:** 1× NVIDIA A100 (40 GB).
- **Llama3-70B configuration:** 4× NVIDIA H100 (94 GB each), tensor parallelism TP = 2.
- The host must have the NVIDIA driver and [`nvidia-container-toolkit`](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed to run the container with `--gpus all`.

The experiments are expected to take approximately:

| Experiment | Approx. runtime |
|---|---|
| Experiment 1 (Security / Table 1, Figure 6) | TODO |
| Experiment 2 (Figure 3, exploitability) | TODO |
| Experiment 3 (Figure 7, cross-workload) | TODO |
| Experiment 4 (Figure 8, cross-model) | TODO |
| Experiment 5 (Figure 9, request-rate sensitivity) | TODO |
| Experiment 6 (Figure 10, threshold sensitivity) | TODO |

Runtime may vary depending on GPU model, model sizes evaluated, and system load.

### Dependencies

All software dependencies are pre-installed in the Docker image; no host-level installation is required beyond Docker and the NVIDIA container runtime. Model weights are not bundled in the image due to size/licensing and must be downloaded separately (see below).

---

## 🚀 Getting Started

The following instructions provide the shortest path for an evaluator to verify that the artifact has been installed correctly.

### 1. Pull the image

```bash
docker pull panapenn/cachesolidarity:latest
```

### 2. Extract the required files from the Docker image

Create the local directory and copy `run.sh` and `plots.ipynb` from the Docker image:

```bash
mkdir -p ~/cachesolidarity_run/models

docker create --name temp_container panapenn/cachesolidarity:latest

docker cp temp_container:/app/run.sh ~/cachesolidarity_run/run.sh
docker cp temp_container:/app/plots.ipynb ~/cachesolidarity_run/plots.ipynb

docker rm temp_container
```

### 3. Set up Hugging Face

Create a free Hugging Face account: https://huggingface.co/join

Accept the license for each gated model used in the evaluation:

- https://huggingface.co/meta-llama/Llama-2-13b-chat-hf
- https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
- https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct
- https://huggingface.co/google/gemma-3-4b-it

Click **"Agree and access repository"** on each repository.

Install and authenticate the Hugging Face CLI:

```bash
pip install huggingface_hub
huggingface-cli login
```

### 4. Download the required models

Download all required models directly into the local `models/` directory:

```bash
huggingface-cli download meta-llama/Llama-2-13b-chat-hf \
  --local-dir ~/cachesolidarity_run/models/Llama-2-13b-chat-hf

huggingface-cli download meta-llama/Llama-2-7b-chat-hf \
  --local-dir ~/cachesolidarity_run/models/Llama-2-7b-chat-hf

huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --local-dir ~/cachesolidarity_run/models/Llama-3.2-1B-Instruct

huggingface-cli download google/gemma-3-4b-it \
  --local-dir ~/cachesolidarity_run/models/gemma-3-4b-it

huggingface-cli download llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
  --local-dir ~/cachesolidarity_run/models/llava-onevision-qwen2-0.5b-ov-hf

huggingface-cli download Qwen/Qwen2.5-VL-3B-Instruct \
  --local-dir ~/cachesolidarity_run/models/Qwen2.5-VL-3B-Instruct
```

After completing the setup, the directory should contain:

```
~/cachesolidarity_run/
├── run.sh
├── plots.ipynb
└── models/
    ├── Llama-2-13b-chat-hf/
    ├── Llama-2-7b-chat-hf/
    ├── Llama-3.2-1B-Instruct/
    ├── gemma-3-4b-it/
    ├── llava-onevision-qwen2-0.5b-ov-hf/
    └── Qwen2.5-VL-3B-Instruct/
```

### 5. Run the container

Once the required models have been downloaded into `~/cachesolidarity_run/models`, run:

```bash
docker run --gpus all \
  --shm-size=16g \
  -v ~/cachesolidarity_run/models:/app/models \
  -v ~/cachesolidarity_run/temp_results:/app/temp_results \
  -v ~/cachesolidarity_run/run.sh:/app/run.sh \
  panapenn/cachesolidarity:latest
```

The local `models/` directory is mounted to `/app/models` inside the container, so the container can access all downloaded models. Results are written to `~/cachesolidarity_run/temp_results` on the host. Because `run.sh` is mounted from the host, you can edit it (e.g., to remove models, workloads, or request rates) to run a reduced evaluation without rebuilding the image.

`run.sh` performs pre-flight verification on startup: it checks that every required model is present under `./models/<name>/config.json` and aborts with a clear error naming any missing model.

---

## 🔬 Reproducing the Results

The generated figures will be in `./plots`. By comparing those with the paper figures as described in [Relationship between the artifact and the paper](#relationship-between-the-artifact-and-the-paper), you can complete the evaluation of this artifact.

```
Pull image & mount models
          ↓
Run run.sh (Phase 1: security pipeline)
          ↓
Run run.sh (Phase 2: performance sweeps)
          ↓
Generate figures and tables (plots.ipynb)
          ↓
Compare with paper
          ↓
Done!
```

---

## 🎯 Claims Supported by the Artifact

This section maps the main claims of the paper to the experiments in the artifact. Experiments can also be run in batch via `run.sh`, as explained above, and directly compared to the figures of the paper without running each block separately.

### Claim 1 — Security of Selective Isolation

**Paper claim:** CacheSolidarity secures the large majority of real-world prompts containing secrets (96.12%) across four datasets, closely matching the semantic-aware SafeKV baseline (96.95%), without relying on semantic analysis.

**Artifact support:** Evaluated using the Table-1/Table-2 and PII-detection blocks of `run.sh` (`pii_masking.py` → `bert_pii_rank.py` → `safeKV_privacy_eval.py`).

**Expected result:** Up to ~76% of secrets detected per dataset by the rule-based/LLM-based pipeline; CacheSolidarity secures ≈96% of prompts, matching SafeKV, with failures concentrated in first-entry/first-try edge cases.

**Corresponding paper result:** Table 1, Figure 6, Section 4.1/5.2.5.

### Claim 2 — Conditional Exploitability of APC Timing Leaks

**Paper claim:** The distinguishability of cache-hit/miss timing depends on prefix length, model size, request rate, and GPU hardware; the side channel weakens under high load and strengthens for larger models and longer prefixes.

**Artifact support:** Evaluated using the Figure-3 block of `run.sh` (and `run_70b.sh` for Llama3-70B).

**Expected result:** KDE overlap decreases (side channel strengthens) with longer prefixes and larger models, and increases (side channel weakens) with higher request rate.

**Corresponding paper result:** Figure 3, Section 2.2.

### Claim 3 — Performance Preservation Across Workloads

**Paper claim:** CacheSolidarity stays within 5–10% of the insecure Prefix Caching baseline's TTFT and hit rate across workloads, while significantly outperforming User Cache Isolation and SafeKV.

**Artifact support:** Evaluated using the Fig-7 block of `run.sh`.

**Expected result:** CacheSolidarity tracks Prefix Caching closely (within ~6% TTFT) while User Cache Isolation and SafeKV show markedly lower hit rate / higher TTFT.

**Corresponding paper result:** Figure 7, Section 5.2.1.

### Claim 4 — Performance Preservation Across Models

**Paper claim:** CacheSolidarity's performance advantage over the baselines holds consistently across LLM families and sizes.

**Artifact support:** Evaluated using the Fig-8 block of `run.sh` (and `run_70b.sh` for Llama3-70B).

**Expected result:** Consistent hit-rate/TTFT ordering (Prefix Caching ≈ CacheSolidarity > SafeKV > User Cache Isolation) across all evaluated models.

**Corresponding paper result:** Figure 8, Section 5.2.2.

### Claim 5 — Robustness to Request Rate

**Paper claim:** CacheSolidarity's hit rate remains stable and close to Prefix Caching as request rate increases, unlike SafeKV, whose hit rate degrades under load.

**Artifact support:** Evaluated using the Fig-9 block of `run.sh`.

**Expected result:** CacheSolidarity's hit rate stays flat as RPS increases; SafeKV's hit rate degrades and its TTFT rises more sharply.

**Corresponding paper result:** Figure 9, Section 5.2.3.

### Claim 6 — Security–Performance Trade-off via the Activator

**Paper claim:** Increasing the Activator's threshold θ monotonically reduces attack success rate, with adjusting θ sufficient to fully prevent the evaluated prompt-stealing attack, at the cost of reduced hit rate/increased TTFT.

**Artifact support:** Evaluated using the Fig-10 block of `run.sh`.

**Expected result:** Attack success rate falls to 0%, with a corresponding gradual increase in TTFT and decrease in hit rate as θ grows.

**Corresponding paper result:** Figure 10, Section 5.3.

---

## Output files

Results are saved under `./temp_results/<experiment_name>/`.

After completing all experiments, this directory should include (among others):

| Output | Description |
|---|---|
| `privacy_evaluation/eval_data/` | Masked and ranked PII-evaluation datasets. |
| `pii_results/` | Per-dataset PII-detection/recovery results (Table 1, Figure 6). |
| `Figure_3_exploitability/` | Raw TTFT logs across models/request rates (Figure 3). |
| `fig7_workloads/` | Hit rate and TTFT logs across workloads and baselines (Figure 7). |
| `fig8_models/` | Hit rate and TTFT logs across models (Figure 8). |
| `fig9_RPS/` | Hit rate and TTFT logs across request rates (Figure 9). |
| `fig10_theta_performance/` | KDE overlap, attack success rate, hit rate, and TTFT logs across thresholds (Figure 10). |

### Generate the figures

Open `plots.ipynb` (next to `temp_results/`) and run all cells. The resulting figures will be generated at `./plots`.

**Expected result:** The generated plots should reproduce the qualitative and quantitative trends shown in Figures 3, 6–10 and Table 1 of the paper. Small numerical differences may occur due to hardware differences and workload sampling randomness.

---

## Data Sources & Disclaimer

This artifact uses real user prompts from public datasets for academic research, reference, and evaluation purposes only.

### External Data References

- **ShareGPT, Flan, UltraChat, OASST1:** publicly available conversational/instruction-tuning datasets, processed with [ai4privacy](https://huggingface.co/ai4privacy) and a BERT-based classifier to identify and mask personally identifiable information.

> **Disclaimer:** All datasets are used solely to evaluate the exploitability and mitigation of timing side channels in shared prefix caches. No secrets or personally identifiable information from these datasets are redistributed as part of this artifact beyond what is necessary to reproduce the paper's aggregate statistics.

---

## 📜 License

This artifact is distributed under the Apache-2.0 license. The bundled vLLM code is also licensed under Apache-2.0 (see `vllm/LICENSE`).


---

## 🙏 Acknowledgements

The work by the authors at the IMDEA Software Institute was partially funded by the Madrid Regional Government through the César Nombela grant (2024-T1/COM-31302) and by the Comunidad de Madrid through the DATIA project, co-funded by the European Union's FEDER funds. Their work was also supported by grant PID2022-142290OB-I00, funded by MCIN/AEI/10.13039/501100011033 and FEDER, UE, and by grant CEX2024-001471-M, funded by MICIU/AEI/10.13039/501100011033. The work by J.O.I is partially supported by the EU EDGELESS project funded by the EU programme (agreement No. 101092950) and by the Smart Networks and Services Joint Undertaking (SNS JU) under the EU HE programme (agreement No. 101293102).

---

## 🔗 Links

- **Paper:** TODO
- **DOI:** TODO
- **Docker Hub:** https://hub.docker.com/r/panapenn/cachesolidarity
- **Middleware 2026:** https://middleware-conf.github.io/2026/
