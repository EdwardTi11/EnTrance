# EnTrance: Entropy-Guided Neural Traversal for Adaptive Decoding

🚀 **Dynamic inference-time decoding for autoregressive language models.**

**EnTrance** is an inference-time decoding framework that monitors a language model's token distribution at every decoding step to dynamically adapt the sampling temperature on the fly.

Instead of applying a rigid, fixed temperature across an entire generation, EnTrance tracks the model's normalized entropy against a rolling historical window. It automatically lowers temperature when the model is confident and raises it when the model encounters complex, ambiguous branching points.

The implementation runs directly on model logits using `llama.cpp` and `llama-cpp-python`.

---

## 💡 Core Concept

At each generation step, EnTrance hooks into the model's logit outputs and observes three core distribution properties:

- **Entropy:** How broadly probability is scattered across the vocabulary.
- **Margin:** The delta between the top two highest logits.
- **Top-k Concentration:** The collective probability mass contained in the top k tokens.

```text
logits ──> probability distribution ──> observations ──> adaptive policy ──> sampling
```

By calculating an **entropy z-score** against a rolling history window (default: 64 steps), the system continuously tunes the temperature:

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
                                                    Rolling History
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
- **BigCodeBench**

To execute the benchmark suite:

```bash
python evaluate.py
```

*Note: The model is programmatically reset before every single evaluation generation snippet to guarantee unbiased data states.*

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
