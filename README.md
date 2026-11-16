# Layer-Optimization-Leveraging
# Layer Optimization Leveraging CrossCoT-QLoRA-Qwen3 with LIME Explainability

**Disentangling content plausibility from formal validity in multilingual syllogistic reasoning — SemEval-2026 Task 11 (Subtask 3).**

![Python](https://img.shields.io/badge/python-3.10-blue)
![PyTorch](https://img.shields.io/badge/pytorch-2.10%2Bcu128-ee4c2c)
![GPU](https://img.shields.io/badge/GPU-1%C3%97NVIDIA%20T4%2016GB-76b900)
![Paper](https://img.shields.io/badge/ICISET-2026-00457c)
![Task](https://img.shields.io/badge/SemEval%E2%80%932026-Task%2011%20%C2%B7%20Subtask%203-orange)
[![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.XXXXXXX-blue)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Official code and data for:

> **Layer Optimization Leveraging CrossCoT-QLoRA-Qwen3 Framework with LIME Explainability for Multilingual Disentangling Content and Formal Reasoning in LLMs**
> Abdullah Al Fahad, Sanjit Sharif, Md Solaiman Islam Shadhin, Md. Ashikuzzaman Bijoy, Sabekun Nahar Chowdhury, Radiathun Tasnia
> Dept. of CSE, Southeast University, Dhaka-1208, Bangladesh
> *2026 5th Int. Conf. on Innovations in Science, Engineering and Technology (ICISET)*, Chittagong, Bangladesh, 8–10 Nov 2026.

---

## Contents

- [The 30-second version](#the-30-second-version)
- [Headline results](#headline-results)
- [What the claim is](#what-the-claim-is)
- [Repository structure](#repository-structure)
- [Quickstart](#quickstart)
- [Reproducing every number in the paper](#reproducing-every-number-in-the-paper)
- [Data](#data)
- [Method](#method)
- [Configuration reference](#configuration-reference)
- [Results to paper table map](#results-to-paper-table-map)
- [Hardware, runtime and environment](#hardware-runtime-and-environment)
- [Gotchas](#gotchas-read-before-you-run)
- [Known issues and errata](#known-issues-and-errata)
- [Citation](#citation)
- [License and acknowledgements](#license-and-acknowledgements)

---

## The 30-second version

Large language models conflate **formal validity** (does the conclusion follow?) with **content plausibility** (is it believable?) — the belief-bias effect. We turn this into a single **4-class** joint prediction problem

```
{valid, invalid} × {plausible, implausible}
```

and attack it with four ingredients: **Cross-CoT prompting** (English pivot instructions + 12 cross-lingual exemplars), **4-bit NF4 QLoRA via Unsloth** on Qwen3, a **softmax-weighted layer-span pooling** module that learns *which* transformer layers to read from, and a **DenseNet bottleneck head**.

The finding that gives the paper its name: **how deep you pool matters more than how big your model is.** Pooling the **last 8 layers** beats last-10, last-12 and full-depth pooling on *both* benchmarks, while costing **33% fewer pooling FLOPs** than last-12. **LIME** shows why: at depth 8 the attribution mass sits on quantifiers and connectives (`all`, `some`, `no`, *therefore*, *consequently*); at full depth it collapses onto entity nouns (*feather*, *stones*) — the model reverts to content heuristics.

Everything trains on **one free Google-Colab T4 (16 GB)**.

---

## Headline results

Primary metric is **macro-averaged F1** over the 4 joint classes.

### Layer-span ablation (paper Table V)

| Benchmark | Pooling span | Macro F1 | Macro Prec. | Macro Rec. / Acc. | Pooling FLOPs |
|---|---|---:|---:|---:|---:|
| **English** (Qwen3-4B) | **−1 … −8 (K=8)** | **0.9524** | 0.9532 | 0.9527 / 0.9529 | **40,960** |
| | −1 … −10 | 0.9468 | 0.9474 | 0.9473 / 0.9476 | 51,200 |
| | −1 … −12 | 0.9424 | 0.9457 | 0.9425 / 0.9424 | 61,440 |
| | Full (−1 … −36) | 0.8846 | 0.8907 | 0.8890 / 0.8901 | 184,320 |
| **Multilingual** (Qwen3-1.7B) | **−1 … −8 (K=8)** | **0.9582** | 0.9598 | 0.9585 / 0.9583 | **32,768** |
| | −1 … −10 | 0.9529 | 0.9532 | 0.9531 / 0.9531 | 40,960 |
| | −1 … −12 | 0.9529 | 0.9532 | 0.9531 / 0.9531 | 49,152 |
| | Full (−1 … −28) | 0.9373 | 0.9388 | 0.9385 / 0.9375 | 114,688 |

FLOPs are the head-side pooling cost, computed as **2 · K · d** MACs: `K` pooled layers × hidden size `d` × 2 ops per multiply–accumulate. `8 × 2 × 2560 = 40,960` (English); `8 × 2 × 2048 = 32,768` (multilingual). K=8 vs K=12 is exactly **33%** fewer: `1 − 40,960/61,440 = 0.333`.

Monotone degradation with depth on English (0.9524 → 0.9468 → 0.9424 → 0.8846) is the core result: **content saturation grows with pooling depth.**

### Per class at K=8 (paper Table IV)

| Benchmark | Class | Prec. | Rec. | F1 |
|---|---|---:|---:|---:|
| English | Valid & Plausible | 0.9787 | 0.9583 | 0.9684 |
| English | Valid & Implausible | 0.9216 | 0.9792 | 0.9495 |
| English | Invalid & Plausible | 0.9556 | 0.9149 | 0.9348 |
| English | Invalid & Implausible | 0.9583 | 0.9583 | 0.9583 |
| Multilingual | Valid & Plausible | 0.9800 | 0.9600 | 0.9700 |
| Multilingual | Valid & Implausible | 0.9400 | 0.9800 | 0.9600 |
| Multilingual | Invalid & Plausible | 0.9100 | 0.9300 | 0.9200 |
| Multilingual | Invalid & Implausible | 0.9800 | 0.9400 | 0.9600 |

No class falls below 0.92 F1. **Invalid & Plausible** is hardest everywhere (subtle structural invalidity inside believable content); **Valid & Implausible** is the cleanest — the strongest single-class evidence that belief bias is suppressed.

### Comparison with related work (paper Table VI)

| Method | Macro F1 |
|---|---:|
| Symbolic + Structural normalisation | 0.9800 |
| **CrossCoT-FT + Qwen3 + last 8 layers (ours)** | **0.9524** |
| Various LLMs + Activation Steering | 0.9100 |
| GPT-4, zero-shot | 0.8200 |
| PaLM-2 540B, few-shot | 0.8150 |
| PaLM 540B, few-shot | 0.7600 |

Prior systems report **binary** validity or plausibility; the joint 4-class task requires resolving both axes at once and is strictly harder. Symbolic/structural methods score higher but rely on explicit representational interventions (deterministic parsing, FOL provers) that CrossCoT-FT does not use.

### Per-language breakdown, multilingual

Reported for the **K=10** run, which is the configuration committed in `notebooks/` (`docs/REPRODUCIBILITY.md` has the K=8 row).

| Lang | n | Macro F1 | Acc. |
|---|---:|---:|---:|
| de | 18 | 1.0000 | 1.0000 |
| ru | 17 | 1.0000 | 1.0000 |
| te | 17 | 1.0000 | 1.0000 |
| pt | 17 | 0.9495 | 0.9412 |
| es | 18 | 0.9451 | 0.9444 |
| fr | 18 | 0.9444 | 0.9444 |
| nl | 17 | 0.9365 | 0.9412 |
| bn | 17 | 0.9143 | 0.9412 |
| sw | 17 | 0.9000 | 0.9412 |
| zh-CN | 18 | 0.8889 | 0.9444 |
| it | 18 | 0.8816 | 0.8889 |

The English-pivot CrossCoT prompt transfers axis separation to Germanic, Slavic, Romance, Indo-Aryan, Sino-Tibetan and Niger-Congo families **without translated scaffolding**. Bengali, Telugu and Swahili (agglutinative / low-resource) remain the weak tail.

---

## What the claim is

Precisely, so it can be checked:

1. **Layer-span depth is a first-class hyperparameter.** Contiguous last-K softmax-weighted pooling with K=8 dominates K∈{10,12} and full depth on both benchmarks.
2. **Accuracy and cost align.** K=8 uses 33% fewer pooling FLOPs than K=12 while scoring higher — the optimum is not a cost trade-off.
3. **The effect is mechanistic, not just numeric.** LIME attribution moves from logical operators (K=8) to entity nouns (full depth), and multilingual confidence collapses from 92.99% → 56.55% when the full stack is pooled.
4. **One T4 is enough.** 6.1 GB peak VRAM for Qwen3-4B, ~78 min per English run.

**Not claimed:** state of the art on binary validity (deterministic/symbolic systems are higher); anything about non-syllogistic reasoning; anything beyond Qwen3-1.7B/4B.

---

## Repository structure

```
.
├── README.md                     this file
├── LICENSE
├── CITATION.cff                  citation metadata (feeds GitHub + Zenodo)
├── requirements.txt              pinned to the exact T4 environment used for the paper
├── environment.yml
├── Makefile                      make setup | lint | audit | ablation | tables
│
├── configs/                      one YAML per reported run — only `layer_span` varies
│   ├── english/{qwen3-4b_last8,last10,last12,full36}.yaml
│   └── multilingual/{qwen3-1.7b_last8,last10,last12,full28}.yaml
│
├── data/
│   ├── README.md                 provenance, licence and checksums for every file
│   ├── raw/                      immutable SemEval-2026 Task 11 release (never edited)
│   ├── provided/                 committed model inputs
│   │   ├── english/train.csv           1,278 rows
│   │   ├── english/test.csv              191 rows
│   │   ├── multilingual/train.csv      1,000 rows
│   │   ├── multilingual/test.csv         192 rows, unlabelled (as released)
│   │   └── multilingual/test_gold.csv    192 rows, + gold labels
│   └── generated/                gitignored: tokenised caches, folds, predictions
│
├── notebooks/                    numbered by execution order, outputs stripped
│   ├── 00_colab_setup.ipynb
│   ├── 01_data_audit.ipynb       leakage + label/language audit (runs in seconds, no GPU)
│   ├── 02_english_qwen3-4b_layer_span.ipynb
│   ├── 03_multilingual_qwen3-1.7b_finetune.ipynb
│   ├── 04_multilingual_layer_tune.ipynb
│   ├── 05_lime_explanations.ipynb
│   ├── 06_paper_figures_and_tables.ipynb
│   └── archive/                  the three original notebooks, verbatim
│
├── src/                          reusable package (notebooks import from here)
│   ├── config.py                 dataclass schema + YAML loader
│   ├── data.py                   load → validate → split → leakage audit
│   ├── prompts.py                Cross-CoT builders + 3 TTA variants
│   ├── models/
│   │   ├── pooling.py            SoftmaxLayerPooling  (Eq. 1)
│   │   ├── dense_head.py         DenseResHead
│   │   ├── qwen3_4b.py           English-track classifier wrapper
│   │   └── qwen3_1p7b.py         SeqCLS + LayerTunedSeqCLS wrappers
│   ├── train.py                  single entrypoint: python -m crosscot.train --config …
│   ├── evaluate.py  predict.py
│   └── explain.py (LIME)   report.py (figures + Excel)
│
├── scripts/
│   ├── run_english.sh  run_multilingual.sh
│   ├── run_layer_ablation.sh     loops K ∈ {8,10,12,full} × {en,ml} → results/
│   └── export_paper_tables.py    results/*.csv → paper/tables/*.tex
│
├── results/                      committed — every number, as data not prose
│   ├── english/layer_ablation.csv        per_class_last8.csv
│   ├── multilingual/layer_ablation.csv   per_class_last8.csv
│   │                 per_language_last10.csv   stage1_baseline.csv
│   ├── comparison_related_work.csv
│   ├── metrics/*.json            per-run machine-readable metrics
│   └── predictions/*.csv         per-item predictions + class confidences
│
├── figures/                      fig1 … fig7, regenerable from notebooks/
├── paper/
│   ├── iciset2026-crosscot-ft.pdf
│   ├── tables/                   .tex generated from results/
│   └── README.md                 figure/table ↔ script map
├── tests/                        test_data.py  test_pooling.py  test_prompts.py
└── docs/
    ├── REPRODUCIBILITY.md        exact commands, expected outputs, runtimes
    ├── DATA.md                   schemas, label semantics, statistics
    └── ERRATA.md                 paper ↔ artifact discrepancies
```

Two rules hold throughout: **everything under `data/raw/` is immutable**, and **everything outside `data/raw/`, `data/provided/`, `configs/`, `src/`, `notebooks/` is deletable and regenerable** by running `make all`.

---

## Quickstart

### A. Google Colab (works today, zero install)

1. Open `notebooks/00_colab_setup.ipynb` in Colab and run it (installs, mounts Drive, sets `/content`).
2. Run `notebooks/01_data_audit.ipynb` → uploads `data/provided/**` and prints the leakage/label audit. **Do not skip this**; it exits non-zero if train and test overlap.
3. English track: `notebooks/02_english_qwen3-4b_layer_span.ipynb` — set `NUM_LAST_LAYERS ∈ {8,10,12}`.
4. Multilingual track: `notebooks/03_…finetune.ipynb`, then `notebooks/04_…layer_tune.ipynb` (set `LAYER_RANGE = -8 | -10 | -12`).
5. `notebooks/05_lime_explanations.ipynb`, then `notebooks/06_paper_figures_and_tables.ipynb`.

Use a **T4** runtime (Runtime → Change runtime type → GPU). 6.1 GB peak VRAM for Qwen3-4B, ~78 min/run.

### B. Local / cluster

```bash
git clone https://github.com/Dark-Stellar/Layer-Optimization-Leveraging.git
cd Layer-Optimization-Leveraging

python -m venv .venv && source .venv/bin/activate
make setup                     # pip install -r requirements.txt + pre-commit

make audit                     # data integrity + leakage check (CPU, seconds)

# one run, English, last 8 layers  (~78 min on a T4)
python -m crosscot.train    --config configs/english/qwen3-4b_last8.yaml
python -m crosscot.evaluate --config configs/english/qwen3-4b_last8.yaml
python -m crosscot.explain  --config configs/english/qwen3-4b_last8.yaml --n-samples 50

# the whole Table V ablation, both benchmarks
make ablation                  # 8 runs; see docs/REPRODUCIBILITY.md for wall-clock

# regenerate every table in the paper
make tables                    # results/*.csv → paper/tables/*.tex
```

Every entrypoint takes `--config` and writes to `results/<track>/<config-name>/`. No path is hardcoded inside the code — Colab paths are a default in `configs/_base.yaml`, overridable with `--set data.train_csv=/your/path.csv`.

### C. Inference only

```python
from crosscot import load_model, predict

model, tok = load_model("configs/english/qwen3-4b_last8.yaml",
                        ckpt="results/english/qwen3-4b_last8/best.pt")
preds, probs = predict(
    model, tok,
    ["All cats are animals. No animals are plants. So no cats are plants."],
    tta=True,                  # soft-votes over the 3 prompt variants
)
# → "valid and implausible"
```

Trained weights are **not** in git (the multilingual adapter alone is 1.4 GB). They are published on the Hugging Face Hub; `docs/REPRODUCIBILITY.md` lists the revision and SHA256 for each checkpoint.

---

## Reproducing every number in the paper

| Paper item | Command | Output | Expected |
|---|---|---|---|
| Table II — dataset statistics | `make audit` | `results/dataset_stats.csv` | 1,278 / 191 EN; 1,000 / 192 ML |
| Table III — hyperparameters | `cat configs/**/*.yaml` | — | see [Configuration reference](#configuration-reference) |
| Table IV — per class @ K=8 | `make ablation && make tables` | `results/{english,multilingual}/per_class_last8.csv` | EN F1 0.9684 / 0.9495 / 0.9348 / 0.9583 |
| Table V — layer ablation | `make ablation` | `results/{english,multilingual}/layer_ablation.csv` | EN 0.9524 / 0.9468 / 0.9424 / 0.8846 |
| Table VI — related work | `results/comparison_related_work.csv` (literature values, cited) | — | static table |
| Figure 3 — confusion matrices | `notebooks/06_…` | `figures/fig3_confusion_matrices.pdf` | strong diagonal, no cross-axis conflation |
| Figures 4–7 — LIME | `python -m crosscot.explain --n-samples 50` | `figures/fig4…fig7_lime_*.pdf` | quantifier/connective mass at K=8 |
| Per-language breakdown | `notebooks/04_…` | `results/multilingual/per_language_last10.csv` | see table above |

### Reproduction status

Honest state of each claim against the committed artifacts:

| Claim | Reproduced by | Status |
|---|---|---|
| English K=8 macro F1 **0.9524**, acc 0.9529, 182/191 correct | `notebooks/02_…` (`NUM_LAST_LAYERS=8`) | ✅ exact — loss 0.4915, 78.0 min, 6.13 GB |
| Multilingual K=10 macro F1 **0.9529** (+ per-language table) | `notebooks/03_…` → `04_…` (`LAYER_RANGE=-10`) | ✅ exact |
| Multilingual stage-1 baseline **0.9426** | `notebooks/03_…` | ✅ exact |
| Multilingual K=8 **0.9582** | `notebooks/04_…` with `LAYER_RANGE=-8` | ⚠️ config committed, run not yet archived |
| English K=10 / K=12 / full-36 | `notebooks/02_…` with `NUM_LAST_LAYERS` | ⚠️ configs committed, runs not yet archived |
| LIME figures 4–7 | `notebooks/05_…` | ⚠️ committed notebook used 10 samples; paper reports 50 |

Set `LAYER_RANGE` / `NUM_LAST_LAYERS` from the YAML, not by editing a cell — that is the whole point of `configs/`.

---

## Data

Full documentation, schemas and checksums: **[`docs/DATA.md`](docs/DATA.md)**.

### Provenance

Both benchmarks come from **SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in Large Language Models** [(Valentino et al., 2026)](https://aclanthology.org/2026.semeval-1.450/). Please cite the task paper alongside ours, and respect the task organisers' licence (see `data/README.md`).

| File | Rows | Source | Notes |
|---|---:|---|---|
| `english/train.csv` | 1,278 | Task 11 Subtask 1 train + `synthetic_hard_v2` augmentation (320 rows) | curated, de-duplicated, leakage-audited |
| `english/test.csv` | 191 | Task 11 Subtask 1 validation split | 0 id / 0 text overlap with train |
| `multilingual/train.csv` | 1,000 | 12-language synthetic syllogisms, 49/51 plausible/implausible balance | `syllogism` = English pivot, `syllogism_t` = translation |
| `multilingual/test.csv` | 192 | Task 11 Subtask 3 test, unlabelled as released | 11 language codes |
| `multilingual/test_gold.csv` | 192 | same + gold labels (post-evaluation release) | ids identical to the file above |

### Schema

```csv
id, syllogism, label                          # english/*.csv
id, syllogism, syllogism_t, lang, label       # multilingual/*.csv
```

- `syllogism` — English source text.
- `syllogism_t` — target-language translation; **this is the model input** for the multilingual track.
- `lang` — ISO 639-1 code. See [errata](#known-issues-and-errata) on `zh` vs `zh-CN`.
- `label` — one of exactly four strings:
  `valid and plausible` · `valid and implausible` · `invalid and plausible` · `invalid and implausible`
  → ids `0, 1, 2, 3` (fixed mapping, identical in both tracks, defined once in `src/data.py`).

### Distributions

| Split | Rows | V+PL | V+IM | INV+PL | INV+IM |
|---|---:|---:|---:|---:|---:|
| English train | 1,278 | 320 | 320 | 313 | 325 |
| English test | 191 | 48 | 48 | 47 | 48 |
| Multilingual train | 1,000 | 245 | 245 | 255 | 255 |
| Multilingual test | 192 | 48 | 48 | 46 | 50 |

Multilingual training languages: `it 98, es 94, de 93, te 93, bn 92, fr 90, ru 88, sw 88, pt 86, nl 85, zh 75, zh-CN 18`.
Multilingual test languages: `it, es, fr, zh-CN, de` 18 each; `te, sw, pt, nl, bn, ru` 17 each.

### Leakage audit

`make audit` (equivalently `notebooks/01_data_audit.ipynb`) checks id-level **and** normalised-string-level overlap between every train and test pair and fails loudly on any hit.

- **English: clean** — 0 shared ids, 0 shared syllogisms.
- **Multilingual: the 192 test items are a verbatim subset of `multilingual/train.csv`.** This is a genuine defect in the committed pipeline, disclosed in full in [`docs/ERRATA.md`](docs/ERRATA.md). Treat the multilingual numbers accordingly until the training file is de-contaminated.

---

## Method

Four stages (paper Figure 2):

**1 · Cross-CoT prompting.** Every input is wrapped in a single English-pivot prompt: definitions of the two axes, a 3-step procedure (*(1) is a counterexample possible? → validity; (2) is it believable? → plausibility; (3) combine*), and **12 cross-lingual labelled exemplars** covering all four classes across the 12 languages. For non-English inputs the instructions stay in English while the syllogism body is in the target language — axis separation transfers without translated scaffolding. Three semantically parallel variants (`v1/v2/v3`) enable **test-time augmentation** by soft-voting their probability distributions.

**2 · QLoRA + Unsloth.** 4-bit NF4 quantisation; frozen `W`, trainable `W′ = W + B·A` with `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×d)`, `r ≪ d`. Adapters on all seven projections (`q, k, v, o, gate, up, down`).

**3 · Softmax-weighted layer-span pooling** (the contribution the paper is named for). Instead of reading only `h_L` — which carries peak autoregressive vocabulary-completion signal and therefore peak content saturation — pool the last-token hidden states of a contiguous span `S_K`:

```
z = tanh( Σ_{l∈S_K}  [ exp(a_l/τ) / Σ_{l'∈S_K} exp(a_l'/τ) ] · h_l )      (Eq. 1)
```

`a_l` is a **learnable scalar per layer**, `τ` a fixed temperature. This is a differentiable, learned structural feature selector: the model decides how much of each layer to use. `S_K` is ablated over `K ∈ {8, 10, 12, full}`.

**4 · DenseNet bottleneck head + label smoothing.** `z` → bottleneck → 4 dense blocks (each concatenating all prior feature maps) → transition → 4-way softmax. Loss:

```
L = −(1−ε)·log P(y=y*|S) − (ε/3)·Σ_{k=0}^{3} log P(y=k|S),   ε = 0.05
```

**Explainability.** LIME fits a locally linear surrogate over token perturbations and reports per-token attribution for all four classes. The diagnostic question is whether attribution mass lands on **logical operators** (quantifiers, connectives → structural reasoning) or **entity nouns** (→ residual content-driven reasoning). At K=8: `feather +0.26`, `no +0.12`, `some +0.08`, `Consequently −0.05`, 94.61% confidence; the Russian instance mirrors the hierarchy at 92.99% (`Есть +0.30`, `камни +0.24`, `Нет +0.14`, `Следовательно +0.10`). At full depth on the multilingual instance, confidence collapses to **56.55%**, leaking 23.64% onto *Invalid & Implausible*.

---

## Configuration reference

Paper Table III. **Only `layer_span` varies between ablation conditions** — everything else is held constant, which is what makes the ablation causal.

| Parameter | Qwen3-4B (English) | Qwen3-1.7B (Multilingual) |
|---|---|---|
| Architecture | 36 layers, d = 2560 | 28 layers, d = 2048 |
| Quantisation / loader | 4-bit NF4 / Unsloth | 4-bit NF4 / Unsloth |
| LoRA r / α | 64 / 128 | 16 / 32 |
| LoRA dropout | 0.0, all projections | 0.2, all projections |
| Trainable parameters | 120 M (2.5%) | 17.4 M (1.0%) |
| Extra unfreezing | last-12 LayerNorms + `lm_head` | none |
| LR / scheduler | 1.5 × 10⁻⁴ / cosine (+restarts) | 2 × 10⁻⁴ / cosine |
| Optimiser | AdamW 8-bit | AdamW 8-bit |
| Effective batch / epochs | 16 / 6 | 8 / 3 |
| Warmup / weight decay / clip | 0.10 / 0.01 / 1.0 | 0.10 / 0.01 / 0.3 |
| Max sequence / peak VRAM | 320 tokens / 6.1 GB | 612 tokens / 8.7 GB |
| TTA variants / eval interval | 3 / every 75 steps | 3 / every 50 steps |
| Label smoothing | 0.05 | 0.05 |
| Early stopping | patience 5, threshold 0.001 | patience 8, threshold 0.001 |
| Selection metric | val macro F1 | val macro F1 |

Head: DenseNet bottleneck, hidden 256, 4 dense layers, bottleneck factor 4, dropout 0.05 — identical in both tracks.

> Values above are **as published**. Three of them differ from the committed notebooks; see [`docs/ERRATA.md`](docs/ERRATA.md) before quoting them.

A config file is the complete description of a run:

```yaml
# configs/english/qwen3-4b_last8.yaml
track: english
model:
  name: unsloth/Qwen3-4B
  n_layers: 36
  hidden_size: 2560
  load_in_4bit: true
layer_span: [-1, -2, -3, -4, -5, -6, -7, -8]   # ← the ablation variable
pooling: { mode: weighted, learnable_weights: true, temperature: 1.0 }
lora:
  r: 64
  alpha: 128
  dropout: 0.0
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
unfreeze: { attention_norm: true, last_n_blocks: 12, lm_head: true }
head: { dropout: 0.05, growth_rate: 256, num_dense_layers: 4, bottleneck_factor: 4 }
train:
  lr: 1.5e-4
  scheduler: cosine_with_restarts
  epochs: 6
  per_device_batch: 2
  grad_accum: 8
  warmup_ratio: 0.10
  weight_decay: 0.01
  max_grad_norm: 1.0
  label_smoothing: 0.05
  eval_steps: 75
  patience: 5
  fp16: true
  optim: adamw_8bit
tta:  { enabled: true, variants: 3 }
lime: { n_samples: 50, num_features: 8, num_perturbations: 512 }
data:
  train_csv: data/provided/english/train.csv
  test_csv:  data/provided/english/test.csv
  max_length: 320
  val_ratio: 0.10
  seed: 42
```

---

## Results to paper table map

`paper/README.md` carries the authoritative mapping; `scripts/export_paper_tables.py` renders each `.tex` from the matching CSV, so no number in the camera-ready is typed by hand.

| Paper table | Source of truth | Regenerate with |
|---|---|---|
| Table I — task instances | `docs/DATA.md` | manual |
| Table II — dataset statistics | `results/dataset_stats.csv` | `make audit` |
| Table III — hyperparameters | `configs/**/*.yaml` | `make tables` |
| Table IV — per class @ K=8 | `results/{english,multilingual}/per_class_last8.csv` | `make tables` |
| Table V — layer ablation | `results/{english,multilingual}/layer_ablation.csv` | `make ablation && make tables` |
| Table VI — related work | `results/comparison_related_work.csv` | static (literature) |
| Figure 1 — overview | `figures/fig1_overview.pdf` | manual |
| Figure 2 — framework | `figures/fig2_framework.pdf` | manual |
| Figure 3 — confusion matrices | `notebooks/06_…` | `make figures` |
| Figures 4–7 — LIME | `python -m crosscot.explain` | `make lime` |

---

## Hardware, runtime and environment

Everything in the paper was produced on **a single NVIDIA T4 (16 GB)** — a free Google Colab GPU.

| Run | Wall clock | Peak VRAM | Trainable params | Final loss |
|---|---:|---:|---:|---:|
| English Qwen3-4B, K=8 | 78.0 min | 6.13 GB | 137.3 M (5.19% of 2.65 B) | 0.4915 |
| Multilingual Qwen3-1.7B, stage 1 | 88.4 min | 2.20 GB | 17.44 M (1.66%) | 0.6396 |
| Multilingual layer-tune, K=10 | ~88 min | ~2.2 GB | 2.36 M (0.22%) | — |

Environment (exact): Python 3.10 · PyTorch 2.10.0+cu128 · CUDA 12.8 · Transformers 5.5.0 · Unsloth 2026.6.9 (English) / 2026.5.4 (multilingual) · Triton 3.6.0 · bitsandbytes · PEFT · `trl<0.9.0` · scikit-learn · lime · openpyxl. Full pin set in `requirements.txt`; `pip freeze` of the exact paper runs is archived in `docs/REPRODUCIBILITY.md`.

`SEED = 42` is set for `random`, `numpy` and `torch` (CPU + CUDA) at the top of every entrypoint. Splits are stratified on `label_id`.

---

## Gotchas (read before you run)

Non-obvious things that cost us hours:

- **Gradient checkpointing must stay OFF in the multilingual custom wrapper.** `gradient_checkpointing=True` creates hook conflicts with 4-bit quantised layers inside `UnslothForSeqCLS` / `LayerTunedSeqCLS`. Memory is managed with small batch + grad accumulation instead.
- **Tied weights break safetensors.** `embed_tokens` and `lm_head` share memory, so `save_safetensors=True` refuses to write two keys pointing at the same tensor. The English track saves a raw `state_dict` as `pytorch_model.pt`; the multilingual track forces `.bin`. Both are handled in `src/train.py`.
- **`num_items_in_batch` warning is expected.** Custom `PreTrainedModel` subclasses do not accept it; gradient accumulation becomes very slightly less accurate. Not a bug.
- **Never return `hidden_states` from `forward()`.** Doing so OOMs the Trainer during evaluation, because it tries to gather and concatenate 37 × batch × seq × 2560 tensors across accumulation steps.
- **Sequence length is tight.** English prompt worst case is 274 tokens against a 320 limit; multilingual Cross-CoT is ~404 tokens against 612. `01_data_audit.ipynb` asserts the worst case — re-run it if you edit the prompts.
- **Padding side must be `right`**, and the pooled vector is taken from the **last real token** (`attention_mask.sum(1) - 1`), not `[CLS]`.
- **Do not commit weights.** The multilingual adapter is 1.4 GB in `.safetensors` and 1.4 GB in `.pt`. Publish to the Hub; keep the URL and SHA256 in the repo.
- **Notebook outputs are stripped** via `nbstripout` (see `.gitattributes`). Without this, one notebook edit produces a 500 KB diff.

---

## Known issues and errata

Disclosed in full in [`docs/ERRATA.md`](docs/ERRATA.md). Summary:

1. **Multilingual train/test contamination (material).** All 192 multilingual test rows appear verbatim in `multilingual/train.csv` (identical id, English text, translation, language and label); 160 fall in the training split under the committed seed. The English track is unaffected. Until `multilingual/train.csv` is de-contaminated, the multilingual macro-F1 figures should be read as optimistic.
2. **Row counts.** Table II reports English test = 192 (48/48/46/50); the committed file has **191** rows (48/48/47/48). The 192/48-48-46-50 distribution is the multilingual test set.
3. **Split ratios.** Table II reports multilingual 800/200; the notebooks use **850/150** (15% validation).
4. **Parameter counts.** Table III reports 120 M (2.5%) for Qwen3-4B; the run logs print 137,303,564 — 5.19% of 2.65 B, or 3.30% of the 4.16 B uncompressed total.
5. **Three multilingual hyperparameters.** Table III says 3 epochs / clip 0.3 / 8.7 GB; the notebook ran 11 epochs / clip 1.0 / 2.20 GB peak.
6. **Language codes.** `zh` and `zh-CN` are the same language split across two codes (train: `zh` 75 + `zh-CN` 18). Twelve *codes*, eleven *languages*; `en` is absent because it is the pivot column.
7. **LIME sample size.** The paper reports 50 explained instances per benchmark; the committed English notebook used 10.
8. **PDF filename** still carries an earlier draft title ("Layer_Span_Feature_Selection…"); the accepted title is "Layer Optimization Leveraging…".

We will correct these in the next revision. If you are building on this work, please read `docs/ERRATA.md` first.

---

## Citation

If you use this code, data or the layer-span pooling idea, please cite the paper **and** the SemEval-2026 Task 11 organisers.

```bibtex
@inproceedings{fahad2026layer,
  title     = {Layer Optimization Leveraging {CrossCoT}-{QLoRA}-{Qwen3} Framework with
               {LIME} Explainability for Multilingual Disentangling Content and
               Formal Reasoning in {LLMs}},
  author    = {Al Fahad, Abdullah and Sharif, Sanjit and Shadhin, Md Solaiman Islam and
               Bijoy, Md. Ashikuzzaman and Chowdhury, Sabekun Nahar and Tasnia, Radiathun},
  booktitle = {2026 5th International Conference on Innovations in Science,
               Engineering and Technology (ICISET)},
  address   = {Chittagong, Bangladesh},
  month     = nov,
  year      = {2026},
  publisher = {IEEE},
  doi       = {10.1109/ICISET.2026.XXXXXXX}
}
```

```bibtex
@inproceedings{valentino2026semeval,
  title     = {SemEval-2026 Task 11: Disentangling Content and Formal Reasoning
               in Large Language Models},
  author    = {Valentino, Marco and Ranaldi, Leonardo and Pucci, Giulia and
               Ranaldi, Federico and Freitas, Andr{\'e}},
  booktitle = {Proceedings of the 20th International Workshop on
               Semantic Evaluation (SemEval-2026)},
  year      = {2026}
}
```

This repository is archived on Zenodo: **DOI `10.5281/zenodo.XXXXXXX`** — cite the version DOI for the exact code you used (see `CITATION.cff`).

---

## License and acknowledgements

Code: **MIT** (see `LICENSE`). Data: **SemEval-2026 Task 11** — original licence retained in `data/README.md`; cite the task paper before redistributing.

**Authors** — Dept. of Computer Science & Engineering, Southeast University, Dhaka-1208, Bangladesh:
Abdullah Al Fahad · Sanjit Sharif · Md Solaiman Islam Shadhin · Md. Ashikuzzaman Bijoy · Sabekun Nahar Chowdhury · Radiathun Tasnia
Contact: `2023100000289@seu.edu.bd` (Abdullah Al Fahad)

**Acknowledgements.** The SemEval-2026 Task 11 organisers for the benchmark; Unsloth AI for the 4-bit training stack that made single-T4 Qwen3-4B fine-tuning feasible; the Qwen team; and the LIME authors. All experiments ran on Google Colab T4 GPUs.

Contributions welcome — see `docs/REPRODUCIBILITY.md` for how to add a run and `tests/` for what must keep passing.