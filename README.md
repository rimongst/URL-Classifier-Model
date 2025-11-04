## Changelog — 04/11/2025

**Updated script:** `urlbert_trainer.py`
**Scope:** Completes optimization (feature engineering/GBDT interaction + ensemble upgrade). 

---

### 1) Feature Engineering & GBDT Interaction Improvements

* **Native categorical handling (LightGBM)**

  * Replaced One-Hot Encoding for categorical fields (e.g., `location`, `extension`) with **label encoding** and passed them to LightGBM via `categorical_feature`.
  * **Why:** Avoids dimensionality blow-up from OHE, enables category-aware splits, and speeds up training.
  * **Effect:** Lower feature dimensionality, faster convergence, typically better generalization.

* **Dimensionality reduction for URLBERT embeddings**

  * Previously used last-layer/last-4-layer pooled embeddings (≈1.5–1.9k dense dims).
  * Applied **TruncatedSVD/PCA** to compress BERT embeddings to **128–256 dims** (configurable in code).
  * **Why:** Reduces noise and overfitting risk; improves training speed and memory footprint.
  * **Pipeline note:** Dense numeric features remain dense; sparse TF-IDF (if enabled) remains sparse. OHE columns for the above categoricals are removed from the stacked matrix.

* **Artifacts**

  * Persisted **label encoders** for categoricals and the **BERT embedding reducer** (e.g., `svd_bert.joblib`) alongside model binaries for reproducible inference.

---

### 2) Ensemble Upgrade — From Blending to Stacking

* **Base learners:** LightGBM and XGBoost trained under **GroupKFold**.
* **OOF predictions:** For each fold, generate **out-of-fold (OOF)** class probabilities; concatenate across folds to create full-length OOF matrices (`N × C`) for each base model.
* **Meta-model:** Train a **stacking head** (default: Logistic Regression; optional: small LightGBM) on `[OOF_LGBM | OOF_XGB]`.
* **Inference path:** Base models → predict probabilities → concatenate → meta-model → final probabilities/predictions.
* **Why:** Allows the meta-model to learn conditional trust (e.g., “prefer LGBM when confident; otherwise defer to XGB”), outperforming fixed global weight blends.
* **Compatibility:** Legacy fixed-weight blending is deprecated (kept as a fallback behind a flag if present).

---

### Breaking Changes

* **Retraining required:** Previous artifacts (with OHE categoricals and raw high-dim BERT embeddings) are **not compatible**.
* **New saved artifacts per run:**

  * Label encoders for categorical fields.
  * Embedding reducer for BERT (e.g., `svd_bert.joblib`).
  * Base model binaries (LGBM/XGB) **and** the stacking meta-model.
* **Prediction pipeline changed:** Inference now routes through the **stacked meta-model** instead of a fixed-weight blend.

---

### Migration Notes

* Remove/ignore prior OHE mappings for the affected categorical columns.
* Fit **label encoders** and the **BERT reducer** on the **training split only**; apply `.transform()` to validation/test to prevent leakage.
* Keep the **GroupKFold grouping key** (e.g., domain/site) stable across runs for fair evaluation.
* Re-export and version all preprocessing objects with the model bundle to ensure deterministic inference.

---

### Rationale & Expected Impact

* **Native categorical splits** (LightGBM) capture category structure better than OHE and reduce feature dimensionality.
* **Embedding compression** (SVD/PCA) preserves salient signal while improving training speed and reducing memory.
* **Stacking** learns **when** to trust each base model, typically improving macro-F1 and robustness versus global blending.

---

### Files Touched

* `urlbert_trainer.py` — training pipeline, preprocessing hooks (categorical label encoding & BERT reducer), OOF generation, stacking meta-model training, and updated inference path.
