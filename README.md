## ? Project Goal

This project studies the behavior of different optimizers on a small Transformer under controlled settings.

We aim to answer:

> How do different optimizers behave under:
>
> 1. a synthetic, fully controlled task
> 2. a small real-text language modeling task

---

## ?? Experimental Setup

### Model

* Small decoder-only Transformer (~10M parameters)

### Optimizers compared

* AdamW
* SGD (with momentum)
* RMSprop (and variants)
* Adafactor

### Controlled variables

* Same model architecture
* Same dataset (per experiment)
* Same batch size
* Same training steps
* Same initialization

### Tuned variable

* Learning rate (separately tuned for each optimizer per task)

---

## ? Experiment 1: Synthetic Task

### Task

* Random token sequence
* Next-token prediction: `y = roll(x)`

### Key property

* True data distribution is uniform
* Theoretical lower bound:
  [
  \text{loss} = \ln(\text{vocab size})
  ]

### Observation

* All optimizers converge close to the same lower bound
* Differences between optimizers are small
* RMSprop / SGD can even slightly outperform AdamW

### Interpretation

This task is **too simple and well-conditioned**, so:

* Optimization is easy
* All reasonable optimizers can reach near-optimal solutions
* Differences mainly reflect minor scaling behavior

---

## ? Experiment 2: Real Text (Harry Potter, ~20k chars)

### Task

* Character-level next-token prediction on real text

### Key differences from synthetic task

* Non-uniform token distribution
* Strong local structure (words, punctuation)
* More complex gradient statistics

### Results (1000 steps, tuned learning rates)

| optimizer | final_loss | avg_step_time |
| --------- | ---------- | ------------- |
| **AdamW** | **1.975**  | 0.0277        |
| Adafactor | 2.380      | 0.0469        |
| SGD       | 2.468      | **0.0256**    |
| RMSprop   | 2.635      | 0.0259        |

---

## ? Key Findings

### 1. Optimizer differences are task-dependent

* On synthetic data: differences are small
* On real text: differences become significant

> Optimizer rankings **do not transfer directly** from synthetic to real tasks.

---

### 2. AdamW performs best on real text

* Achieves the lowest final loss
* Especially strong in early and mid training

Interpretation:

* Combines:

  * first-moment momentum (stable direction)
  * second-moment scaling (adaptive step size)
* Better suited for Transformer + real data distributions

---

### 3. RMSprop performs well on synthetic but poorly on real data

* Works well when:

  * gradient statistics are simple
* Struggles when:

  * data distribution is complex
  * gradient structure is uneven

Interpretation:

> Per-coordinate scaling alone is not sufficient;
> directional smoothing (momentum) becomes important.

---

### 4. SGD is fast but less effective

* Lowest step time
* Higher final loss

Interpretation:

> Lacks adaptive scaling ¡ú inefficient in heterogeneous parameter space

---

### 5. Adafactor trades performance for structure

* Slower and slightly worse than AdamW
* No benefit at this small scale

Interpretation:

> Designed for large-scale memory efficiency, not small-model optimization quality

---

## ? Overall Insight

> The effectiveness of an optimizer strongly depends on the **data distribution and gradient structure**, not just the algorithm itself.

* Synthetic tasks ¡ú mask differences
* Real tasks ¡ú amplify differences

---

## ?? Limitations

* Small model (~10M parameters)
* Small dataset (~20k characters)
* Single-seed results (future work: multi-seed averaging)

---

## ? Future Work

* Multi-seed experiments (mean ¡À std)
* Longer training horizon
* Larger model / sequence length
* More realistic datasets

