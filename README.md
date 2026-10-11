# EnTrance: Entropy-Guided Neural Traversal for Adaptive Decoding

🚀 **Dynamic inference-time decoding for autoregressive language models.**

**EnTrance** is an inference-time decoding framework that monitors a language model's token distribution at every decoding step to dynamically adapt the sampling temperature on the fly.

Instead of applying a rigid, fixed temperature across an entire generation, EnTrance tracks the model's normalized entropy using an incremental Exponential Moving Average (EMA). It automatically lowers temperature when the model is confident and raises it when the model encounters complex, ambiguous branching points.

The implementation runs directly on model logits using `llama.cpp` and `llama-cpp-python`.

---

## 💡 Core Concept

At each generation step, EnTrance hooks into the model's logit outputs and observes core distribution properties:

- **Entropy:** How broadly probability is scattered across the full vocabulary.
- **Margin:** The gap between the top-1 and top-2 logits ($p_{\text{top1}} - p_{\text{top2}}$).
- **Min-P Candidate Dynamics:** The candidate token count (`min_p_count`) and cumulative probability mass (`min_p_mass`) surviving the dynamic `min_p` cutoff $(p \ge \mathtt{min\_p} \times p_{\mathtt{max}})$

```text
logits ──> probability distribution ──> observations ──> EMA tracker ──> adaptive policy ──> sampling
```

By calculating an **entropy z-score** via a streaming Exponential Moving Average (EMA) ($O(1)$ constant memory and time updates), the system continuously tunes the temperature:

- **High Entropy (z > 0):** Unusually high uncertainty → **Higher Temperature** (encourages exploration).
- **Low Entropy (z < 0):** High model confidence → **Lower Temperature** (encourages precision).

---

## 🛠 Architecture Flow

```text
                         Prompt ──> Chat Formatting ──> Tokenize
                                                            │
                                                            ▼
                                                      Language Model
                                                            │
                                                            ▼
                                                          Logits
                                                            │
                                                            ▼
                                                   ┌─────────────────┐
                                                   │ Observe Metrics │
                                                   └────────┬────────┘
                                                            │
                                                            ▼
                                                  Streaming EMA Tracker
                                                   (Entropy z-score)
                                                            │
                                                            ▼
                                                   Decoder Controller
                                                            │
                                                            ▼
                                                   Adaptive Temperature
                                                            │
                                                            ▼
                                                  Top-k / Top-p Sampling
                                                            │
                                                            ▼
                                                    Selected Token ──> Loop Next Step
```

---

## ⚙️ Project Structure

```text
EnTrance/
├── logs/ # Results from benchmarks from evaluate.py
├── model_design/
│   ├── adaptive_control.py  # Metrics tracking & z-score temperature scaling
│   └── engine.py            # Token-by-token generation & custom sampler
├── main.py                  # Interactive CLI chat interface
├── evaluate.py              # Benchmark runner (Inspect AI)
├── models/                  # Local GGUF model storage
└── requirements.txt         # Dependencies
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Ensure you have your C++ compilers set up properly for `llama-cpp-python` compilation acceleration (CUDA, Metal, etc.).

```bash
# Clone the repository
git clone https://github.com
cd EnTrance

# Install dependencies
pip install -r requirements.txt
```

### 2. Download a Model

Place any compatible `.gguf` model into the `models/` directory:

```bash
mkdir -p models/
# Download your preferred GGUF into models/ (e.g., Llama-3, Mistral, Qwen)
```

### 3. Run Interactive Chat

Test the adaptive decoding framework natively via the command line interface:

```bash
python main.py
```

---

## 📊 Evaluation & Benchmarking

EnTrance features a robust evaluation framework powered by **Inspect AI** to compare standard fixed-parameter decoding (Baseline) against entropy-adaptive decoding.

The harness natively supports three complex reasoning benchmarks:

- **AIME 2025**
- **GPQA Diamond**

To execute the benchmark suite:

```bash
python evaluate.py
```

*Note: The model is programmatically reset before every single evaluation generation snippet to guarantee unbiased data states.*

---

## 📈 Results

Initial evaluation compared fixed-parameter **Baseline** decoding with **EnTrance** (entropy-adaptive temperature) on shuffled samples from AIME 2025 and GPQA Diamond.

### Completion-token usage

| Benchmark | Mode | Samples processed (reported) | Total completion tokens | Avg. completion tokens / sample |
|---|---|---:|---:|---:|
| AIME 2025 | Baseline | 10 | 40,390 | 4,039.0 |
| AIME 2025 | EnTrance | 10 | 40,522 | 4,052.2 |
| GPQA Diamond | Baseline | 40 | 142,944 | 3,573.6 |
| GPQA Diamond | EnTrance | 40 | 145,432 | 3,635.8 |

### Constraints

Evaluations were run locally on vanilla llama.cpp using CPU thread-pooling without hardware acceleration (such as CUDA or Vulkan). The resulting computational bottlenecks and latency constraints limited the sample sizes per benchmark during this initial prototyping phase.

### Observations

- **AIME 2025:** EnTrance generated 132 more completion tokens overall, an increase of approximately **0.33%** over the baseline. Average completion length increased by **13.2 tokens per sample**.
- **GPQA Diamond:** EnTrance generated 2,488 more completion tokens overall, an increase of approximately **1.74%** over the baseline. Average completion length increased by **62.2 tokens per sample**.
- **Prompt-token usage was unchanged** between modes in both reported runs: 2,432 tokens for AIME 2025 and 5,160 tokens for GPQA Diamond.

These initial results show that adaptive decoding used slightly more completion tokens on both benchmarks. **They do not establish an improvement or regression in answer quality**: accuracy, benchmark scores, latency, and computational cost were not included in the supplied output. Those metrics should be reported in future comparisons before drawing conclusions about reasoning performance or efficiency.

> **Reporting note:** The GPQA section header says “10 samples,” while the run output reports 40 samples processed. Inspect AI runs the model through GPQA questions in 4 epochs for each question. The table uses the processed count reported by the output;

---

## 🎯 Scope Limitations

EnTrance operates entirely via **local logit manipulation on token-by-token processing loops**.

- **Included:** Real-time logit observation, z-score statistical scaling, dynamic temperature selection, and comprehensive trace logging.
- **Excluded:** This framework does **not** implement energy-based search graphs, beam search variations, or lookahead predictive branching thresholds.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

EnTrance sits on top of exceptional open-source ecosystems:

- [**llama.cpp**](https://github.com/ggerganov/llama.cpp) & [**llama-cpp-python**](https://github.com/abetlen/llama-cpp-python)
- [**Inspect AI**](https://github.com/UKGovernmentBEIS/inspect_ai) by the UK AI Safety Institute

---
