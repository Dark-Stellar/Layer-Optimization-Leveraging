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
- [Outputs written by each notebook](#outputs-written-by-each-notebook)
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

and attack it with four ingredients: **Cross-CoT prompting** (English pivot instructions + cross-lingual labelled exemplars), **4-bit NF4 QLoRA via Unsloth** on Qwen3, a **softmax-weighted layer-span pooling** module that learns *which* transformer layers to read from, and a **DenseNet bottleneck head**.

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

Reported for the **K=10** run — the configuration stored in `Multilangual_layer_tune_and_evaluate _Final-10.ipynb`.

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

The repository mirrors the two experimental tracks, plus the paper.

```
Layer-Optimization-Leveraging/
│
├── README.md                          this file
│
├── English model/                     Track 1 — English, Qwen3-4B
│   ├── Code/
│   │   └── Eng last 8, 10, 12,full layered tuned.ipynb
│   │         End-to-end: install → HF login → model → train → TTA inference
│   │         → confusion matrix → LIME → Excel report → save .pt to Drive.
│   │         Single knob: NUM_LAST_LAYERS = 8 | 10 | 12  (stored run: 8)
│   └── Data/
│       ├── qwen3_syllogism_train_CURATED_AUG_NONLEAKY.csv   1,278 rows
│       └── test_data_with_labels.csv                          191 rows
│
├── Multilingual model/                Track 2 — 11 languages, Qwen3-1.7B (2 stages)
│   ├── Code/
│   │   ├── multilangual_Qwen_Fine_tune_Final.ipynb
│   │   │     STAGE 1 — QLoRA fine-tune + UnslothForSeqCLS head.
│   │   │     Saves the LoRA adapter (.safetensors / .bin / .pt) to Drive.
│   │   └── Multilangual_layer_tune_and_evaluate _Final-10.ipynb
│   │         STAGE 2 — reloads the Stage-1 .pt from Drive, attaches a new
│   │         head, unfreezes a layer span, re-trains, evaluates, reports.
│   │         Single knob: LAYER_RANGE = -8 | -10 | -12  (stored run: -10)
│   └── Data/
│       ├── train_data_49_51_ratio.csv                       1,000 rows
│       ├── test_data_multilang_CSV.csv                        192 rows (unlabelled)
│       └── test_data_subtask_3_with_labels.csv                192 rows (+ gold labels)
│
└── Paper/
    └── Layer_Span_Feature_Selection_for_Disentangling_Content_and_Formal_Reasoning_in_LLMs__A_CrossCoT_QLoRA_Qwen3_Framework_with_LIME_Explainability.pdf
          ICISET 2026 camera-ready, 6 pages.
```

All three notebooks are **Colab notebooks with their outputs committed** — you can read the exact logged run (metrics, per-class report, per-language F1, timings, VRAM) without executing anything.

> **Note on paths.** The notebooks read their CSVs from `/content/…` (the Colab working directory) and write results to Google Drive. Upload the `Data/` files to `/content/` before running, or edit the three path constants at the top of each notebook. See [Quickstart](#quickstart).

---

## Quickstart

### 1. Prerequisites

- Google Colab (or any single CUDA GPU with ≥ 16 GB). Select **Runtime → Change runtime type → GPU → T4**.
- A Hugging Face token (`hf_…`) for the English track only — the notebook calls `login()` via `getpass`. The Qwen3 multilingual notebooks load `unsloth/Qwen3-1.7B-unsloth-bnb-4bit` without a token.
- ~15 GB free Google Drive for checkpoints (the multilingual adapter is 1.4 GB; do **not** commit weights to git).

### 2. English track — Qwen3-4B

1. Upload `English model/Data/qwen3_syllogism_train_CURATED_AUG_NONLEAKY.csv` and `English model/Data/test_data_with_labels.csv` to `/content/`.
2. Open `English model/Code/Eng last 8, 10, 12,full layered tuned.ipynb` in Colab.
3. Run all cells. When prompted, paste your HF token.
4. To reproduce an ablation condition, change **one line** near the top before running:

   ```python
   NUM_LAST_LAYERS = 8     # ← 8, 10 or 12.  LAYER_INDICES is auto-generated from it.
   ```

The notebook prints the leakage audit, the token-length audit, the per-layer index mapping, the trainable-parameter count, then validation and test reports, writes the confusion matrix, LIME figures and an Excel workbook, and copies everything to `/content/drive/MyDrive/SemEval2026/Qwen3_4B_Unsloth/<RUN_DATE>/`.

### 3. Multilingual track — Qwen3-1.7B (two stages)

**Stage 1** — `multilangual_Qwen_Fine_tune_Final.ipynb`
1. Upload `train_data_49_51_ratio.csv`, `test_data_multilang_CSV.csv` and `test_data_subtask_3_with_labels.csv` to `/content/`.
2. Run all. It trains, predicts on the 192 test items, evaluates against the gold file, prints the per-language F1 table, and saves the adapter to `/content/drive/MyDrive/SemEval2026/92model/multilang_4class_unsloth/lora_adapter/`.

**Stage 2** — `Multilangual_layer_tune_and_evaluate _Final-10.ipynb`
1. Make sure Stage 1 finished and `…/lora_adapter/pytorch_model.pt` exists on Drive (the path is hard-coded in Cell 2 as `SAVED_MODEL_PATH`).
2. Adjust the layer-tuning knobs in **Cell 5**:

   ```python
   LAYER_RANGE              = -10      # -8 / -10 / -12  → last N transformer layers
   PROJECTIONS_TO_UNFREEZE  = ["q_proj", "v_proj", "up_proj", "down_proj"]
   HIDDEN_LAYER_SIZE        = 1024     # extra hidden layer in the new head
   UNFREEZE_ATTENTION_NORM  = True
   UNFREEZE_LM_HEAD         = False
   DENSE_LAYER_SIZE         = 256
   PROJECTION_RATE          = 0.2      # dropout inside the new head
   LT_LEARNING_RATE         = 2e-4
   LT_MAX_EPOCHS            = 11
   LT_PATIENCE              = 8
   ```

3. Run all. It rebuilds the backbone, reloads the Stage-1 weights, attaches the new head, unfreezes only the requested parameters, re-trains, then predicts, evaluates, reports per-language F1 and saves to `/content/drive/MyDrive/SemEval2026/layer_tuned-10/…`.

> **Stage 2 re-runs the Stage-1 split** (`train_test_split(test_size=0.15, random_state=42, stratify=label_id)`), so it is self-contained apart from the adapter file. It also re-downloads the base model — it does **not** rely on any other Stage-1 state.

---

## Reproducing every number in the paper

| Paper item | How to get it | Expected |
|---|---|---|
| Table II — dataset statistics | `pandas.read_csv(...).shape` on each file; see [Data](#data) | 1,278 / 191 EN; 1,000 / 192 ML |
| Table III — hyperparameters | Constants at the top of each notebook; see [Configuration reference](#configuration-reference) | — |
| Table IV — per class @ K=8 | English notebook with `NUM_LAST_LAYERS = 8`, final report cell | EN F1 0.9684 / 0.9495 / 0.9348 / 0.9583 |
| Table V — layer ablation | English notebook × {8,10,12,full}; Stage-2 notebook × {-8,-10,-12,full} | EN 0.9524 / 0.9468 / 0.9424 / 0.8846 |
| Table VI — related work | Literature values as cited in the paper | static table |
| Figure 3 — confusion matrices | English notebook `[7/9]`; Stage-2 notebook Cell 10 | `confusion_matrix_test_<date>.png`, `confusion_matrix.png` |
| Figures 4–7 — LIME | English notebook `[8/9]` → `lime_figures/lime_idx*.png` | quantifier/connective mass at K=8 |
| Per-language breakdown | Stage-1 notebook Cell 10; Stage-2 notebook Cell 11 | see table above |

### Reproduction status

Honest state of each claim against the **committed notebook outputs**:

| Claim | Notebook + setting | Status |
|---|---|---|
| English K=8 macro F1 **0.9524**, acc 0.9529, 182/191 correct | `Eng last 8, 10, 12,full layered tuned.ipynb`, `NUM_LAST_LAYERS = 8` | ✅ exact match with stored outputs (loss 0.4915, 78.0 min, 6.13 GB) |
| Multilingual stage-1 baseline **0.9426** | `multilangual_Qwen_Fine_tune_Final.ipynb` | ✅ exact (loss 0.6396, 5,302 s, 2.20 GB) |
| Multilingual K=10 macro F1 **0.9529** + per-language table | `Multilangual_layer_tune_and_evaluate _Final-10.ipynb`, `LAYER_RANGE = -10` | ✅ exact (2,364,164 trainable params) |
| Multilingual K=8 **0.9582** | same notebook, `LAYER_RANGE = -8` | ⚠️ not archived — one-line change + re-run |
| English K=10 / K=12 / full-36 | `NUM_LAST_LAYERS = 10 / 12` | ⚠️ not archived — one-line change + re-run |
| Multilingual K=12 / full-28 | `LAYER_RANGE = -12` / full | ⚠️ not archived |
| LIME figures 4–7 | English notebook | ⚠️ stored run used 10 explained instances; paper reports 50 (`LIME_N_SAMPLES`) |

Everything needed for the missing rows is a single variable change away — this is the cheapest thing to complete.

---

## Data

### Provenance

Both benchmarks come from **SemEval-2026 Task 11: Disentangling Content and Formal Reasoning in Large Language Models** [(Valentino et al., 2026)](https://aclanthology.org/2026.semeval-1.450/). Please cite the task paper alongside ours, and respect the task organisers' licence.

| File | Rows | Source | Notes |
|---|---:|---|---|
| `English model/Data/qwen3_syllogism_train_CURATED_AUG_NONLEAKY.csv` | 1,278 | Task 11 Subtask 1 train + `synthetic_hard_v2` augmentation (320 rows, id-prefixed `synthetic_hard_v2-`) | curated, de-duplicated, leakage-audited |
| `English model/Data/test_data_with_labels.csv` | 191 | Task 11 Subtask 1 validation split | 0 shared ids and 0 shared strings with train |
| `Multilingual model/Data/train_data_49_51_ratio.csv` | 1,000 | 11-language synthetic syllogisms, 49/51 plausible/implausible balance | `syllogism` = English pivot, `syllogism_t` = translation |
| `Multilingual model/Data/test_data_multilang_CSV.csv` | 192 | Task 11 Subtask 3 test, unlabelled as released | 11 language codes |
| `Multilingual model/Data/test_data_subtask_3_with_labels.csv` | 192 | same rows + gold labels (post-evaluation release) | ids identical to the file above |

### Schema

```csv
id, syllogism, label                          # English model/Data/*.csv
id, syllogism, syllogism_t, lang, label       # Multilingual model/Data/*.csv
```

- `syllogism` — English source text. This is the model input for the **English** track.
- `syllogism_t` — target-language translation. This is the model input for the **multilingual** track.
- `lang` — ISO 639-1 code. See [errata](#known-issues-and-errata) on `zh` vs `zh-CN`.
- `label` — one of exactly four strings:
  `valid and plausible` · `valid and implausible` · `invalid and plausible` · `invalid and implausible`
  → ids `0, 1, 2, 3`. The mapping is identical in both notebooks; the English loader also accepts a `gold_label` column as an alias for `label`.

### Distributions

| Split | Rows | V+PL | V+IM | INV+PL | INV+IM |
|---|---:|---:|---:|---:|---:|
| English train | 1,278 | 320 | 320 | 313 | 325 |
| English test | 191 | 48 | 48 | 47 | 48 |
| Multilingual train | 1,000 | 245 | 245 | 255 | 255 |
| Multilingual test | 192 | 48 | 48 | 46 | 50 |

Multilingual training languages: `it 98, es 94, de 93, te 93, bn 92, fr 90, ru 88, sw 88, pt 86, nl 85, zh 75, zh-CN 18`.
Multilingual test languages: `it, es, fr, zh-CN, de` 18 each; `te, sw, pt, nl, bn, ru` 17 each.

English syllogisms average 127 characters (max 234); multilingual test items 134 (max 228).

### Leakage audit

The English notebook audits leakage at load time and **raises** if any id or syllogism string is shared between train and test:

- **English: clean** — 0 shared ids, 0 shared syllogisms.
- **Multilingual: the 192 test items are a verbatim subset of `train_data_49_51_ratio.csv`.** All 192 rows match on id, English text, translation, language **and** label. Under the notebooks' split (`test_size=0.15, random_state=42, stratify=label_id`), **160** of them land in the training split and 32 in validation. This is a genuine defect in the committed pipeline — see [errata](#known-issues-and-errata) item 1.

---

## Method

Four stages (paper Figure 2):

**1 · Cross-CoT prompting.** Every input is wrapped in a single English-pivot prompt: definitions of the two axes, a 3-step procedure (*(1) is a counterexample possible? → validity; (2) is it believable? → plausibility; (3) combine*), and **12 cross-lingual labelled exemplars** covering all four classes. For non-English inputs the instructions stay in English while the syllogism body is in the target language — axis separation transfers without translated scaffolding. The English track additionally defines three semantically parallel variants (`v1/v2/v3`, differing in system wording and the verb before the syllogism) and applies **test-time augmentation** by soft-voting their probability distributions.

**2 · QLoRA + Unsloth.** 4-bit NF4 quantisation; frozen `W`, trainable `W′ = W + B·A` with `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×d)`, `r ≪ d`. Adapters on all seven projections (`q, k, v, o, gate, up, down`).

**3 · Softmax-weighted layer-span pooling** (the contribution the paper is named for). Instead of reading only `h_L` — which carries peak autoregressive vocabulary-completion signal and therefore peak content saturation — pool the last-token hidden states of a contiguous span `S_K`:

```
z = tanh( Σ_{l∈S_K}  [ exp(a_l/τ) / Σ_{l'∈S_K} exp(a_l'/τ) ] · h_l )      (Eq. 1)
```

`a_l` is a **learnable scalar per layer**, `τ` a fixed temperature. This is a differentiable, learned structural feature selector: the model decides how much of each layer to use. `S_K` is ablated over `K ∈ {8, 10, 12, full}`.

Implementation note: the English notebook realises Eq. 1 with a `layer_weights` `nn.Parameter` passed through `softmax`, indexing `hidden_states[total + li]` for negative `li`, and gathering the last real token via `attention_mask.sum(1) - 1`. The multilingual Stage-2 notebook realises the same span idea by **unfreezing** the selected layers' projections (`q_proj`, `v_proj`, `up_proj`, `down_proj`) plus their attention norms — i.e. span selection there is expressed as trainable-parameter selection rather than as a pooling weight vector.

**4 · DenseNet bottleneck head + label smoothing.** `z` → bottleneck → dense blocks (each concatenating all prior feature maps) → transition → 4-way softmax. Loss:

```
L = −(1−ε)·log P(y=y*|S) − (ε/3)·Σ_{k=0}^{3} log P(y=k|S),   ε = 0.05
```

**Explainability.** LIME fits a locally linear surrogate over token perturbations and reports per-token attribution for all four classes. The diagnostic question is whether attribution mass lands on **logical operators** (quantifiers, connectives → structural reasoning) or **entity nouns** (→ residual content-driven reasoning). At K=8: `feather +0.26`, `no +0.12`, `some +0.08`, `Consequently −0.05`, 94.61% confidence; the Russian instance mirrors the hierarchy at 92.99% (`Есть +0.30`, `камни +0.24`, `Нет +0.14`, `Следовательно +0.10`). At full depth on the multilingual instance, confidence collapses to **56.55%**, leaking 23.64% onto *Invalid & Implausible*.

---

## Configuration reference

Paper Table III. **Only the layer span varies between ablation conditions** — everything else is held constant, which is what makes the ablation causal.

| Parameter | Qwen3-4B (English) | Qwen3-1.7B (Multilingual) |
|---|---|---|
| Architecture | 36 layers, d = 2560 | 28 layers, d = 2048 |
| Quantisation / loader | 4-bit NF4 / Unsloth `FastLanguageModel` | 4-bit NF4 / Unsloth `FastModel` |
| LoRA r / α | 64 / 128 | 16 / 32 |
| LoRA dropout | 0.0, all 7 projections | 0.2, `target_modules="all-linear"` |
| Trainable parameters | 120 M (2.5%) | 17.4 M (1.0%) |
| Extra unfreezing | last-12 LayerNorms + `lm_head` | none (Stage 1); layer span + attn-norm (Stage 2) |
| LR / scheduler | 1.5 × 10⁻⁴ / cosine_with_restarts | 2 × 10⁻⁴ / cosine |
| Optimiser | AdamW 8-bit | AdamW 8-bit |
| Effective batch / epochs | 16 / 6 | 8 / 3 |
| Warmup / weight decay / clip | 0.10 / 0.01 / 1.0 | 0.10 / 0.01 / 0.3 |
| Max sequence / peak VRAM | 320 tokens / 6.1 GB | 612 tokens / 8.7 GB |
| TTA variants / eval interval | 3 / every 75 steps | 3 / every 50 steps |
| Label smoothing | 0.05 | 0.05 |
| Early stopping | patience 5, threshold 0.001 | patience 8, threshold 0.001 |
| Selection metric | val macro F1 (`f1_macro`) | val macro F1 (`f1`) |

Head: DenseNet bottleneck, growth rate 256, 4 dense layers, bottleneck factor 4, dropout 0.05 (English); Stage-2 multilingual head uses hidden 1024 / dense 256 / dropout 0.2.

> Values above are **as published**. Several differ from the constants actually used in the committed notebooks; see [errata](#known-issues-and-errata) items 4 and 5 before quoting them.

### The knobs that actually matter

**English** (`CFG` dict, plus `NUM_LAST_LAYERS`):

```python
NUM_LAST_LAYERS   = 8                    # ← the ablation variable
LORA_R, LORA_ALPHA, LORA_DROPOUT = 64, 128, 0.0
POOL_MODE         = "weighted"
UNFREEZE_ATTN_NORM, UNFREEZE_LM_HEAD, UNFREEZE_LAST_N_BLOCKS = True, True, 12
HEAD_DROPOUT, GROWTH_RATE, NUM_DENSE_LAYERS, BOTTLENECK_FACTOR = 0.05, 256, 4, 4
LABEL_SMOOTHING   = 0.05
LEARNING_RATE, MAX_EPOCHS, PATIENCE = 1.5e-4, 6, 5
PER_DEVICE_BATCH, GRAD_ACCUM, WARMUP_RATIO = 2, 8, 0.10
USE_TTA, TTA_VARIANTS = True, 3
USE_LIME, LIME_N_SAMPLES, LIME_NUM_FEATURES, LIME_NUM_PERTURB = True, 10, 8, 512
```

**Multilingual Stage 2** (Cell 5): `LAYER_RANGE`, `PROJECTIONS_TO_UNFREEZE`, `HIDDEN_LAYER_SIZE`, `UNFREEZE_ATTENTION_NORM`, `UNFREEZE_LM_HEAD`, `DENSE_LAYER_SIZE`, `PROJECTION_RATE`, `LT_LEARNING_RATE`, `LT_MAX_EPOCHS`, `LT_PATIENCE`, `TEMPERATURE = 0.9`.

`SEED = 42` is set for `random`, `numpy` and `torch` (CPU + CUDA) at the top of every notebook, and both splits are stratified on `label_id`.

---

## Outputs written by each notebook

**English** — everything goes to `/content/qwen3_4b_<YYYYMMDD_HHMM>/CFG_Last<N>_OrigLoRA_SavePT/` and is mirrored to Drive:

| File | Contents |
|---|---|
| `predictions_test_<date>.csv` | per-item prediction, correctness flag and 4 class confidences |
| `metrics_<date>.json` | val/test macro F1, test accuracy, peak VRAM, runtime, train loss |
| `confusion_matrix_test_<date>.png` | 4×4 test-only confusion matrix (paper Figure 3, left) |
| `lime_figures/lime_idx*.png` | per-instance 4-class LIME attribution plots |
| `results_CFG_Last<N>_…_<date>.xlsx` | 6 sheets: Summary, PerClass, LayerWeights, LIME_Explanations, Architecture, Config |
| `layer_weights_<date>.npy` | learned per-layer softmax weights (the feature selector) |
| `trained_model_best_<date>/pytorch_model.pt` | best checkpoint state dict |

**Multilingual Stage 1** — `/content/semeval_multilang/`: `test_predictions.csv`, `confusion_matrix.png`, `model_performance_report.xlsx`; adapter + tokenizer under `…/92model/multilang_4class_unsloth/lora_adapter/`.

**Multilingual Stage 2** — `/content/semeval_layer_tuned/`: `test_predictions.csv`, `confusion_matrix.png`, `performance_report.xlsx` (Summary + Per_Language); state dict and config under `…/layer_tuned-10/LR<N>_PROJ…/`.

---

## Hardware, runtime and environment

Everything in the paper was produced on **a single NVIDIA T4 (16 GB)** — a free Google Colab GPU.

| Run | Wall clock | Peak VRAM | Trainable params | Final loss |
|---|---:|---:|---:|---:|
| English Qwen3-4B, K=8 | 78.0 min | 6.13 GB | 137,303,564 (5.19% of 2.65 B) | 0.4915 |
| Multilingual Qwen3-1.7B, Stage 1 | 88.4 min (5,302 s) | 2.20 GB | 17,440,772 (1.66%) | 0.6396 |
| Multilingual layer-tune, K=10 | ~88 min | ~2.2 GB | 2,364,164 (0.22%) | — |

Steps: English 432 (71 steps/epoch × 6, effective batch 16); multilingual 1,177 (11 epochs, effective batch 8).

Environment (exact): Python 3.10 · PyTorch 2.10.0+cu128 · CUDA 12.8 · Transformers 5.5.0 · Unsloth 2026.6.9 (English) / 2026.5.4 (multilingual) · Triton 3.6.0 · bitsandbytes · PEFT · `trl<0.9.0` · datasets · scikit-learn · lime · openpyxl · sentencepiece · protobuf.

Dependencies are installed by Cell 1 of each notebook:

```bash
# English
pip install -q unsloth unsloth_zoo
pip install -q "transformers>=4.51.0" peft accelerate scikit-learn seaborn \
               matplotlib openpyxl huggingface_hub lime

# Multilingual (both notebooks)
pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install -q --no-deps "trl<0.9.0" peft accelerate bitsandbytes
pip install -q datasets scikit-learn matplotlib seaborn sentencepiece protobuf
```

---

## Gotchas (read before you run)

Non-obvious things that will bite you:

- **Stage 2 needs Stage 1's `pytorch_model.pt` on Drive.** `SAVED_MODEL_PATH` is hard-coded to `…/92model/multilang_4class_unsloth/lora_adapter/pytorch_model.pt`. If Stage 1 was saved elsewhere, edit that constant or the load will fail.
- **`LAYER_RANGE` is a negative count, not a list.** `-10` means "last 10 layers"; `0` or `None` means head-only tuning.
- **Gradient checkpointing must stay OFF in the multilingual wrappers.** It creates hook conflicts with 4-bit quantised layers inside `UnslothForSeqCLS` / `LayerTunedSeqCLS`; memory is managed with small batch + grad accumulation instead. The English notebook enables it (`use_gradient_checkpointing="unsloth"`) because its wrapper is a plain `nn.Module`.
- **Tied weights break safetensors.** `embed_tokens` and `lm_head` share memory, so `save_safetensors=True` refuses to write two keys pointing at the same tensor. That is why the English trainer saves a raw `state_dict` as `.pt` (custom `PtBinTrainer.save_model`) and why Stage 2 forces `.bin` by monkey-patching `trainer.save_model`.
- **`num_items_in_batch` warning is expected.** Custom `PreTrainedModel` subclasses do not accept it; gradient accumulation becomes very slightly less accurate. Not a bug.
- **Never return `hidden_states` from `forward()`.** Doing so OOMs the Trainer during evaluation (37 × batch × seq × 2560 tensors). The English `prediction_step` is overridden to return only loss/logits/labels.
- **Sequence length is tight.** English prompt worst case is 274 tokens against a 320 limit (the notebook asserts this); multilingual Cross-CoT is ~404 tokens against `MAX_SEQ_LEN = 612`. Re-check if you edit the prompts.
- **Padding side is `right`** and the pooled vector is the **last real token** (`attention_mask.sum(1) - 1`), not `[CLS]` — Qwen3 has no CLS token.
- **Do not commit weights.** The multilingual adapter is 1.4 GB in `.safetensors` and 1.4 GB in `.pt`. Keep them on Drive or the Hugging Face Hub; only the URL and SHA256 belong in git.
- **The two multilingual notebooks duplicate the CoT prompt string.** It must stay byte-identical between Stage 1 and Stage 2 or Stage 2 evaluates a different input distribution than it was tuned for. The comment in Stage 2 (`identical to fine-tune — do not change`) is load-bearing.

---

## Known issues and errata

Discrepancies found by cross-checking the camera-ready against the committed notebooks and data. Item 1 is material; 2–8 are cosmetic.

1. **Multilingual train/test contamination (material).** All 192 rows of `Multilingual model/Data/test_data_subtask_3_with_labels.csv` appear verbatim in `Multilingual model/Data/train_data_49_51_ratio.csv` — identical `id`, English `syllogism`, translated `syllogism_t`, `lang` and `label`. With the committed split, **160 of 192** test items fall in the training split and 32 in validation. The reported multilingual macro F1 (0.9582 @ K=8, 0.9529 @ K=10) is therefore measured partly on data seen during training and should be read as an upper bound. The **English track is clean** (0 shared ids, 0 shared strings), so the central layer-span claim is unaffected. Fix: drop the 192 contaminated rows from the training file and re-run.
2. **Row counts.** Table II reports English test = 192 rows with 48/48/46/50. The committed file has **191** rows with 48/48/47/48. The 192 / 48-48-46-50 line is the *multilingual* test distribution.
3. **Split ratios.** Table II reports multilingual 800/200; both multilingual notebooks use **850/150** (15% validation). English: notebook logs 1,150/128; Table II reports 1,149/129.
4. **Parameter counts.** Table III reports 120 M (2.5%) for Qwen3-4B; the run log prints 137,303,564 — 5.19% of 2.65 B (or 3.30% of the 4.16 B uncompressed total that Unsloth reports).
5. **Three multilingual hyperparameters.** Table III says 3 epochs / grad-clip 0.3 / 8.7 GB; the notebook ran 11 epochs / clip 1.0 / 2.20 GB peak.
6. **Language codes.** `zh` and `zh-CN` are the same language split across two codes (train: `zh` 75 + `zh-CN` 18). Twelve *codes*, eleven *languages*; `en` is absent because English is the pivot/source column, not a translation target.
7. **LIME sample size.** The paper reports 50 explained instances per benchmark; the committed English notebook used `LIME_N_SAMPLES = 10` (80 feature rows).
8. **PDF filename.** The file is named `Layer_Span_Feature_Selection_…` (an earlier draft title); the accepted title is *Layer Optimization Leveraging CrossCoT-QLoRA-Qwen3 Framework with LIME Explainability…*.

We will correct these in the next revision. If you are building on this work, please read this list first.

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

This repository is archived on Zenodo: **DOI `10.5281/zenodo.XXXXXXX`** — cite the version DOI for the exact code you used.

---

## License and acknowledgements

Code: **MIT**. Data: **SemEval-2026 Task 11** — the underlying syllogism data belongs to the task organisers; cite the task paper before redistributing.

**Authors** — Dept. of Computer Science & Engineering, Southeast University, Dhaka-1208, Bangladesh:
Abdullah Al Fahad · Sanjit Sharif · Md Solaiman Islam Shadhin · Md. Ashikuzzaman Bijoy · Sabekun Nahar Chowdhury · Radiathun Tasnia
Contact: `2023100000289@seu.edu.bd` (Abdullah Al Fahad)

**Acknowledgements.** The SemEval-2026 Task 11 organisers for the benchmark; Unsloth AI for the 4-bit training stack that made single-T4 Qwen3-4B fine-tuning feasible; the Qwen team; and the LIME authors. All experiments ran on Google Colab T4 GPUs.

Issues and pull requests welcome — in particular, runs filling the missing rows of Table V are the highest-value contribution.