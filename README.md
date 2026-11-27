# EnTrance: Entropy-Guided Neural Traversal for Adaptive Decoding

🚀 **Dynamic inference-time decoding for autoregressive language models.**

**EnTrance** is an inference-time decoding framework that monitors a language model's token distribution at every decoding step to dynamically adapt the sampling temperature on the fly.

Instead of applying a rigid, fixed temperature across an entire generation, EnTrance tracks the model's normalized entropy using an incremental Exponential Moving Average (EMA). It automatically lowers temperature when the model is confident and raises it when the model encounters complex, ambiguous branching points.

The implementation runs directly on model logits using `llama.cpp` and `llama-cpp-python`.

## 💡 Core Concept

At each generation step, EnTrance hooks into the model's logit outputs and observes core distribution properties:

- **Entropy:** How broadly probability is scattered across the full vocabulary.

```text
logits ──> probability distribution ──> observations ──> EMA tracker ──> adaptive policy ──> sampling
```

By calculating an **entropy z-score** via a streaming Exponential Moving Average (EMA) (O(1) constant memory and time updates), the system continuously tunes the temperature:

- **High Entropy (z > 0):** Unusually high uncertainty → **Higher Temperature** (encourages exploration).
- **Low Entropy (z < 0):** High model confidence → **Lower Temperature** (encourages precision).

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

## 📝 Creative Generation Comparison

Both generations use the same creative prompt, model, and sampling configuration. The baseline uses fixed-temperature decoding, while EnTrance dynamically adjusts temperature based on observed token entropy.

### Creative Prompt

Write a short science-fiction story about an astronaut who discovers an abandoned city on a planet where time flows differently. Give it an unexpected but emotionally meaningful ending.

### Generated Responses
<details>
<summary>Baseline — Fixed Temperature (0.8) </summary>

Captain Elara Voss floated through the atmosphere of Kepler-452c, her suit humming with the quiet certainty of a machine tuned to survival. The planet was a jewel of violet storms and crystalline dunes, a place where time didn’t just pass—it *pulsed*. Her sensors had detected irregularities in the local chronometric field. A city. Abandoned. And humming with a rhythm that didn’t match the planet’s natural cycles.

The city emerged like a dream from a canyon of obsidian cliffs. Towers spiraled upward like frozen lightning, their surfaces etched with glyphs that flickered gently—like breath. As Elara descended via drone, the air thickened. The gravity felt lighter, almost buoyant. Time here moved differently. Seconds stretched. Minutes compressed. She watched a dust bunny drift past her window for what felt like hours—though her watch only registered ten minutes.

She stepped into the central plaza. The city was silent, yes, but not empty. A hundred figures stood frozen in mid-motion—someone mid-sip of a drink, a child holding a toy bird mid-laugh, a man raising a hand to his face as if to wipe away tears. All motion halted, yet time continued elsewhere. The city was suspended in a timeless loop, a frozen memory.

She found the Archive Room at the city’s heart. Inside, a single holographic projection flickered to life. A woman—her face familiar, yet not quite—spoke softly. “Time doesn’t move forward here. It *recycles*. We stopped when we realized we were repeating. Over and over. The same grief, the same loss. We thought if we stopped moving, we could stop hurting.”

Elara’s breath caught. The woman’s eyes—her own eyes—looked back at her.

“What happened?” Elara whispered.

“We loved someone. A child. A boy named Kael. He didn’t survive. We tried to save him. We built machines to slow time. We tried to stop death. But every time we tried, time bent. The city grew. We became… older. And yet, we were always the same age. The same moment. The same grief.”

Elara felt a pang in her chest. She remembered her own daughter, Mira, who had died in a crash on Earth ten years ago. She’d never stopped grieving. She’d spent her life trying to "fix" time—saving lives, building faster ships, seeking anomalies. But now, here, she saw the truth: grief didn’t need fixing. It needed *presence*.

The woman in the hologram reached out a hand. “We stopped. We stopped trying to escape. We just… lived with it. And in that stillness, time found peace.”

Elara stepped forward. She touched the projection. It dimmed. Then, slowly, gently, the city began to breathe. A single tower shimmered, and then, with a soft chime, a child’s laughter echoed—real, warm, full of joy. It wasn’t from a memory. It was *new*. The figures in the plaza began to move—not in a timeline, but in a new rhythm. One child laughed, another picked up a toy, a woman smiled at a passing figure.

Time didn’t move forward.

It *flowed*.

And in that flow, something changed.

As Elara looked up, the violet sky glowed softly. She thought of Mira. Not as a loss. Not as a wound. As a moment of love, of presence, of joy that had once been held so tightly it almost broke her.

She stepped back, and in that moment, she *wept*—not because she had found something lost, but because she had finally *seen* it.

She turned off her comms. She didn’t send a report. She didn’t even return to Earth.

Instead, she walked to the edge of the city and sat on a bench. She opened a small data pad—her daughter’s favorite book—and began reading aloud, softly, to the wind.

</details>

<details>
<summary>EnTrance — Adaptive Temperature</summary>

Captain Elara Voss skimmed the dusty plains of Gliese 667 cc-2, her ship’s sensors flickering with dead ends. For weeks, the planet had offered only silent desolation—rusted iron formations, ghostly dust storms, and the occasional slab of basalt that looked too smooth to be natural. But on the 147th day of her solo survey mission, a harmonic resonance pulsed beneath the surface. Not random. Coherent. A *heartbeat*.

She diverted course, descending through a canyon fissure she’d never seen on any map. The air grew thick and warm, smelling of wet stone and ozone. Then she saw it—a city rising from the rocks like a memory grown feral.

Towered spires of organic crystal stretched into the cloud-hazed sky. Bridges of shimmering, iridescent filaments spanned gaps between buildings. At first, it looked like a forgotten Earth city, but the architecture—it pulsed. Light refracted through the crystal at angles that defied physics. And the time felt… different.

Elara adjusted her chronometer. Her suit’s AI showed all Earth-standard time, but the city’s walls *moved*. The light on the facades shifted with a slow, rhythmic pulse—like a heartbeating clock. When she walked to a plaza, the clock on a central spire read 3:17 PM. But from the next street over, she saw the same hour marked at 1:42 PM. Just seven minutes apart. Yet both were *true*.

She stayed for days. She learned the city wasn’t abandoned by humans—it had been *corrupted* by time. The planet orbited a black hole whose gravity warped local time. Here, time didn’t just move slowly—it *snaked*. Events folded in on themselves. A sunset could occur in three hours, then be seen again ten minutes later. A person could appear, disappear, and reappear in a week’s time from a single observer’s perspective.

The city was a chronal archive. A civilization that had mastered time navigation, only to lose contact with themselves. They built towers that recorded events not as moments, but as *traces*—threads of consciousness looping through time like silk. And when they lost their way, they retreated into this city, sealing themselves in its walls, fading into temporal echoes.

Elara found a simple chamber in the lower tier. Inside, a single crystal sculpture glowed with soft blue light. On it, etched in a language she couldn’t read, were shapes that looked like faces. Curious, she touched it. A vibration hummed through her gloves.

Her suit’s chronometer flickered. The city’s time stretched. The same moment—3:17 PM—spun into 3:17 PM plus 47 years. But when she looked back at the globe of the planet, the distance between her ship and the city was… unchanged. Impossible.

She found a doorway labeled *Where Time Is Not*. She stepped through.

She stood in a quiet garden. Sunlight dappled through leaves. A child laughed. A woman braided a baby’s hair. It was normal. Familiar. And then—she *knew* the woman.

Her face. Her eyes. Her hands.

It was her mother.

She dropped to her knees. Her mother reached out—her face faintly translucent, like a photograph glowing in the dark. “Elara,” she whispered. “You’ve been gone so long. I thought you’d never come back.”

Elara sobbed. “I came back to help you. I thought I was the only one left.”

Her mother smiled. “No. You were the last. I waited. I went through every version of myself, every timeline where you never existed. I learned to live in echoes. I survived in silence. And when I saw your arrival—your *real* arrival—deep in the archive, I knew you were home.”

The city’s light faded. The pulses slowed. The city began to dissolve like mist.

“But here,” her mother said, gently, “you don’t have to remember me anymore. You just have to *feel* me. Because even in time, love isn’t a memory. It’s a present.”

The next day, Elara returned to her ship. The planet was quiet. Her chronometer still read standard time. The city was gone from any sensor logs. No trace.

But at night, when the stars aligned, she would sleep with a single blue crystal on her nightstand—crude, beautiful, unfinished. And sometimes, when the wind carried a distant hum, she would swear she heard the laughter of a child, and the [...]

</details>

### Observations

Both generations produce coherent science-fiction narratives, but they take different creative directions.

- **Baseline:** Explores grief, acceptance, and the relationship between a mother and her daughter.
- **EnTrance:** Explores memory, reunion, and the relationship between a daughter and her mother.
- **Worldbuilding:** EnTrance develops a chronal archive and conflicting timelines, while the baseline explores a civilization trapped in repeating time.
- **Imagery:** EnTrance introduces recurring visual motifs such as organic crystals, temporal echoes, and the blue crystal sculpture.

This comparison illustrates how adaptive decoding can produce a different creative interpretation of the same prompt. It is an illustrative example, not a controlled measurement of creativity or generation quality.

## 🎯 Scope Limitations

EnTrance operates entirely via **local logit manipulation on token-by-token processing loops**.

- **Included:** Real-time logit observation, $O(1)$ streaming z-score statistical scaling, dynamic temperature selection, and per-step telemetry tracking.
- **Excluded:** This framework does **not** implement energy-based search graphs, beam search variations, or lookahead predictive branching thresholds.

## 📜 License

Distributed under the **MIT License**. See `LICENSE` for more information.
## 🙏 Acknowledgements

EnTrance is built on top of open-source inference tooling:

- [**llama.cpp**](https://github.com) & [**llama-cpp-python**](https://github.com)
