# EnTrance

> **Entropy-Guided Neural Traversal for Reasoning and Controlled
> Exploration**

EnTrance is an inference-time reasoning framework that augments
autoregressive language models with **entropy-guided adaptive search**.
Rather than applying expensive search uniformly, EnTrance monitors model
uncertainty and dynamically decides **when additional reasoning is
worthwhile**.

## Features

-   Adaptive entropy-guided inference
-   Dynamic **Adaptive Energy Function**
-   EGALBS (Energy-Guided Adaptive Lookahead Beam Search)
-   Automatic transition between linear decoding and search
-   Online threshold estimation
-   Hyperparameter optimization with Optuna
-   Benchmark suite for Coding, Mathematics, and Logic
-   Compatible with `llama.cpp` GGUF models

------------------------------------------------------------------------

## Design Philosophy

Most generated tokens are straightforward. Only a small fraction
represent critical reasoning decisions.

EnTrance allocates additional computation **only when needed**, using an
adaptive energy threshold rather than performing beam search
continuously.

------------------------------------------------------------------------

## Architecture

``` text
                 User Prompt
                      │
                      ▼
              Language Model
                      │
                      ▼
               Token Logits
                      │
                      ▼
       Adaptive Energy Function
          (Running μ + k · σ)
                      │
             Energy > Threshold?
              ┌────────┴────────┐
              │                 │
             No                Yes
              │                 │
              ▼                 ▼
      Continue Decode      Run EGALBS
                                   │
                      ┌────────────┴────────────┐
                      ▼                         ▼
               Beam Expansion          Lookahead Search
                      │                         │
                      └────────────┬────────────┘
                                   ▼
                      Lowest-Energy Continuation
                                   │
                                   ▼
                         Resume Linear Decoding
                                   │
                                   ▼
                            Final Generation
```

------------------------------------------------------------------------

## Repository Structure

``` text
EnTrance/
├── model_design/
│   ├── energy.py
│   ├── engine.py
│   └── search.py
│
├── testing/
│   ├── prompt_config.py
│   └── tune.py
|
├─ .gitignore
├── models/
├── requirements.txt
├── LICENSE
└── README.md
```

------------------------------------------------------------------------

## Adaptive Energy Function

EnTrance computes an energy score for every decoding step and compares
it against an adaptive threshold derived from recent decoding history.

The trigger threshold is:

``` text
Threshold = μ + kσ
```

Where:

-   **μ** --- Running mean of observed energy values
-   **σ** --- Running standard deviation
-   **k** --- Configurable sensitivity multiplier (`k_multiplier`)

If the current energy remains below the threshold, generation continues
normally.

If the current energy exceeds the threshold, EnTrance temporarily pauses
linear decoding and invokes **EGALBS** before resuming generation.

---------------------------------------------------------------------------

# EGALBS (Energy-Guided Adaptive Lookahead Beam Search)

EGALBS is EnTrance's search algorithm. Unlike traditional beam search, EGALBS is **event-driven** rather than continuously active.

Search is invoked only when the Adaptive Energy Function detects that the model has entered a region of unusually high uncertainty.

The algorithm performs a bounded beam search, evaluates each candidate using cumulative energy, and resumes decoding from the lowest-energy continuation.

---

## Energy Function

``` text
E(x) = -α · log(P(x)) + γ · R(x)
```

Where:

- **E(x)** — Total energy score for token *x*
- **P(x)** — Token surprise (predictive uncertainty component)
- **R(x)** — Repetition penalty component
- **α** (`alpha`) — Weight controlling the contribution of token surprise
- **γ** (`gamma`) — Weight controlling the contribution of repetition penalty

---

## Algorithm

```text
Generate next token
        │
        ▼
Compute token energy
        │
        ▼
Energy > μ + kσ ?
     │          │
     │ No       │ Yes
     ▼          ▼
Continue    Initialize Beam
 Decoding       Search
                   │
                   ▼
        Expand Beam Candidates
                   │
                   ▼
      Perform Lookahead Search
                   │
                   ▼
 Compute Cumulative Energy
                   │
                   ▼
Select Lowest-Energy Path
                   │
                   ▼
Resume Linear Decoding
```

---

## Search Procedure

1. Monitor the energy score after every generated token.
2. Compare the current energy against the adaptive threshold.
3. If the threshold is exceeded:
   - Initialize a beam of candidate continuations.
   - Expand each beam for the configured lookahead depth.
   - Compute cumulative energy for every candidate path.
   - Select the minimum-energy continuation.
4. Append the winning path to the generated sequence.
5. Resume normal linear decoding.

---

## Why EGALBS?

Traditional beam search explores multiple hypotheses throughout the entire decoding process, increasing computational cost even when the model is highly confident.

EGALBS concentrates search only where it is expected to be beneficial.

This provides two advantages:

- **Reduced computation** by avoiding unnecessary search.
- **Improved reasoning quality** by exploring alternative continuations only during periods of elevated uncertainty.

---------------------------------------------------------------------------

## Hyperparameters

  Parameter        Description
  ---------------- ------------------------------------------
  `alpha`          Controls token surprise contribution
  `gamma`          Controls repetition penalty contribution
  `k_multiplier`   Controls search trigger sensitivity

### Fixed Search Parameters

  Parameter           Value
  ----------------- -------
  Beam Width              4
  Lookahead Depth         9

------------------------------------------------------------------------

## Benchmarking

EnTrance evaluates three reasoning domains:

-   Coding
-   Mathematics
-   Logic

Metrics include:

-   Accuracy
-   Total Forward Passes
-   Average Forward Passes

Optuna jointly minimizes:

1.  Errors
2.  Forward passes

------------------------------------------------------------------------

## Installation

``` bash
git clone https://github.com/<username>/EnTrance.git
cd EnTrance
pip install -r requirements.txt
```

Download a supported GGUF model and configure its path by putting it in the models folder before running
benchmarks.

------------------------------------------------------------------------

## License

This project is licensed under the **MIT License**.

See the [LICENSE](LICENSE) file for details.

------------------------------------------------------------------------

## Acknowledgements

EnTrance builds upon:

-   llama.cpp
-   llama-cpp-python
-   NumPy
-   Optuna
