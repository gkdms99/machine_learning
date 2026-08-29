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

![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)
![CatBoost](https://img.shields.io/badge/CatBoost-FFCC00?style=flat-square&logo=catboost&logoColor=black)
![LightGBM](https://img.shields.io/badge/LightGBM-02569B?style=flat-square&logo=lightgbm&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![SHAP](https://img.shields.io/badge/SHAP-8A2BE2?style=flat-square)
![Optuna](https://img.shields.io/badge/Optuna-0077B5?style=flat-square)
![klib](https://img.shields.io/badge/klib-333333?style=flat-square)
![missingno](https://img.shields.io/badge/missingno-333333?style=flat-square)
