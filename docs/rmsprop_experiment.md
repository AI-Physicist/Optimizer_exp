# ? Optimizer Minimal Experiment: RMSprop Ablations

This project investigates the behavior of several optimizers on a minimal synthetic task, with a focus on understanding **what components of adaptive optimizers actually matter**.

In particular, we study:

* RMSprop and its variants
* The role of **EMA (memory)**
* The effect of **norm choice (2-norm vs p-norm)**

---

## ? Task Setup

We use a deliberately simple and controlled task:

* **Task**: next-token prediction with shifted targets
  [
  y = \text{roll}(x)
  ]
* **Vocabulary size**: 32000
* **Theoretical lower bound**:
  [
  \mathcal{L}_{\min} \approx \ln(32000) \approx 10.37
  ]

This provides a clean baseline where:

* Optimization difficulty is low
* Differences between optimizers are easier to isolate

---

## ?? Implemented Optimizers

### Baselines

* SGD
* AdamW
* RMSprop
* Adafactor

### RMSprop Variants (this repo focus)

1. **Standard RMSprop**
   [
   v_t = \beta v_{t-1} + (1-\beta) g_t^2
   ]
   [
   x_{t+1} = x_t - \eta \frac{g_t}{\sqrt{v_t} + \epsilon}
   ]

2. **No-memory RMSprop**
   [
   v_t = g_t^2
   ]

   Equivalent to:
   [
   \Delta x \approx -\eta \cdot \text{sign}(g)
   ]

3. **p-norm RMSprop**

   * Replace 2-norm style scaling with (p)-norm based normalization
   * Used to test sensitivity to normalization geometry

---

## ? How to Run

```bash
cd /data3/hywan/optimizer_exp

python3 train.py --optimizer rmsprop --log_file logs/rmsprop.csv
python3 train.py --optimizer rmsprop_nomem --log_file logs/rmsprop_nomem.csv
python3 train.py --optimizer rmsprop_pnorm --log_file logs/rmsprop_pnorm.csv
```

---

## ? Key Experimental Results

### 1. Removing EMA (Memory)

* Loss curves **quickly overlap within ~1000 steps**
* Only minor differences in early phase

**Interpretation:**

> In this task, RMSprop¡¯s main effect is **per-coordinate normalization**, not temporal smoothing.

---

### 2. Replacing 2-norm with p-norm

* Per-step time **increased by 30%¨C50%**
* Early loss **slightly lower**
* After ~1000 steps: **curves overlap**

**Interpretation:**

> Norm choice affects early optimization trajectory but **does not change long-term behavior**.

---

## ? Core Findings

### ? What actually matters

* **Per-coordinate scaling (preconditioning)** is the dominant factor

### ? What matters less (in this task)

* EMA memory (second-moment smoothing)
* Exact norm used (2-norm vs p-norm)

---

## ? Deeper Interpretation

This experiment suggests:

> RMSprop in this setting behaves more like a **normalized / sign-based optimizer**, rather than a true second-order method.

Specifically:

* Removing EMA ¡ú still works
* Using p-norm ¡ú similar final results
* Implies optimization is driven by:
  [
  \text{direction} \gg \text{precise magnitude}
  ]

---

## ? Why Do Curves Overlap?

Because:

1. Gradient statistics are **stable**
2. Task is **simple and well-conditioned**
3. All variants implement **similar normalization effect**

So:

[
\frac{g}{\sqrt{v}} \approx \frac{g}{|g|}
]

---

## ?? Limitations

These results **do NOT generalize automatically**:

* Small model
* Short training (1000 steps)
* Low-noise gradients
* Synthetic task

In more realistic settings:

* EMA may stabilize noisy gradients
* Norm choice may matter more
* Long-term dynamics may diverge

---

## ? Suggested Future Experiments

To further probe differences:

* [ ] Compare with **signSGD**
* [ ] Reduce batch size ¡ú increase noise
* [ ] Extend training to 10k+ steps
* [ ] Use larger / deeper models
* [ ] Track variance of normalization term

---

## ? Takeaway

> In this minimal setting, RMSprop¡¯s benefit comes mainly from **coordinate-wise normalization**, while more sophisticated design choices (EMA, norm type) have limited impact on final performance.

---

## ? Notes

* This is a **mechanism-level study**, not a benchmark
* Goal: understand *why* optimizers behave differently, not just *which is better*

