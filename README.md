# EnTrance: Entropy-Guided Neural Traversal for Adaptive Decoding

🚀 **Dynamic inference-time decoding for autoregressive language models.**

**EnTrance** is an inference-time decoding framework that monitors a language model's token distribution at every decoding step to dynamically adapt the sampling temperature on the fly.

Instead of applying a rigid, fixed temperature across an entire generation, EnTrance tracks the model's normalized entropy using an incremental Exponential Moving Average (EMA). It automatically lowers temperature when the model is confident and raises it when the model encounters complex, ambiguous branching points.

The implementation runs directly on model logits using `llama.cpp` and `llama-cpp-python`.

---

## 💡 Core Concept

At each generation step, EnTrance hooks into the model's logit outputs and observes core distribution properties:

- **Entropy:** How broadly probability is scattered across the full vocabulary.

```text
logits ──> probability distribution ──> observations ──> EMA tracker ──> adaptive policy ──> sampling
```

By calculating an **entropy z-score** via a streaming Exponential Moving Average (EMA) (O(1) constant memory and time updates), the system continuously tunes the temperature:

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
├── model_design/
│   ├── adaptive_control.py  # Metrics tracking & z-score temperature scaling
│   └── engine.py            # Token-by-token generation & custom sampler
├── models/                  # Local GGUF model storage
├── .gitignore
├── LICENSE                  # MIT License
├── main.py                  # Interactive CLI chat interface
├── pyproject.toml           # Project metadata & dependencies
├── README.md
└── uv.lock                  # Reproducible dependency lockfile
```

---

## 🚀 Quick Start

### 1. Prerequisites & Installation

Ensure you have [uv](https://astral.sh) installed, along with C++ compilers set up properly if building `llama-cpp-python` with hardware acceleration (CUDA, Metal, etc.).

```bash
# Clone the repository
git clone https://github.com/EdwardTi11/EnTrance.git
cd EnTrance

# Create a virtual environment and install dependencies from uv.lock
uv sync
```

*(Optional)* To build `llama-cpp-python` with hardware acceleration before or during sync, pass `CMAKE_ARGS` (for example, `CMAKE_ARGS="-DGGML_CUDA=on" uv sync` or `CMAKE_ARGS="-DGGML_METAL=on" uv sync`).

### 2. Download a Model

Place any compatible `.gguf` model into the `models/` directory:

```bash
mkdir -p models/
# Download your preferred GGUF into models/ (e.g., Llama-3, Mistral, Qwen)
```

### 3. Run Interactive Chat

Test the adaptive decoding framework natively via the command line interface:

```bash
uv run python main.py
```

---

## 🎯 Scope Limitations

EnTrance operates entirely via **local logit manipulation on token-by-token processing loops**.

- **Included:** Real-time logit observation, $O(1)$ streaming z-score statistical scaling, dynamic temperature selection, and per-step telemetry tracking.
- **Excluded:** This framework does **not** implement energy-based search graphs, beam search variations, or lookahead predictive branching thresholds.

---

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.

---

## 🙏 Acknowledgements

EnTrance is built on top of open-source inference tooling:

- [**llama.cpp**](https://github.com) & [**llama-cpp-python**](https://github.com)
