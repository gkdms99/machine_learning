# machine_learning

Modeling notebooks from [KML Challenge 2021F](https://www.kaggle.com/competitions/kml2021f), a binary classification competition on panel survey response data. Ranked 2nd during the internal round — the competition was later opened up publicly, so the current leaderboard reflects a much larger pool of entrants than the one these notebooks competed against.

Final submission scored **0.90603** on the public leaderboard.

## Approach

**Round 1 — `컴피티션_1round_best_score.ipynb`**
- EDA and feature engineering on panel/survey data: response-rate features (question count, response-rate buckets), cumulative reward points, time-of-day/day-of-week response patterns, regex-based title parsing, per-user and per-question response-rate aggregates
- Feature selection via SHAP importance (`shap.TreeExplainer` on an LGBM baseline), thresholding on mean absolute SHAP value
- Modeled as binary classification; compared a tuned DNN (Keras `Dense` layers, widths 8–256, early stopping on validation loss) against gradient-boosted trees

**Round 2 — `2라운드_모델링 및 서브미션 앙상블 (1).ipynb`**
- Re-ran the pipeline separately per month (Jan/Feb/Mar) before and after the round-1 feature transform, using `BaggingClassifier` over `CatBoostClassifier` / `RandomForestClassifier` (5-fold CV) and an early-stopped `LGBMClassifier`
- Feature engineering lifted CV AUC noticeably across all three models (e.g. CatBoost 0.929 → 0.959)
- Final submission blended 6 monthly-averaged model outputs (RF/CatBoost/LGBM, pre- and post-feature-transform) via weighted average, favoring the post-transform CatBoost run (weights 0.1/0.1/0.1/0.4/0.15/0.15)

## Stack

pandas, scikit-learn, CatBoost, LightGBM, Keras/TensorFlow, SHAP, Optuna, klib, missingno
