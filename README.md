# kaggle_aesScore_solution
Automated Essay Scoring (AES) Ensemble Notebooks

This repository contains two Jupyter notebooks that form part of an ensemble solution for the Automated Essay Scoring (AES) task. The goal of this task is to automatically assign a score to an essay. The notebooks utilize a combination of transformer-based models (Deberta-v3-large), LightGBM (LGBM), XGBoost (XGB), and extensive feature engineering, along with a crucial threshold optimization step, to achieve high performance, typically measured by the Quadratic Weighted Kappa (QWK) metric.
Project Structure

    0-818-deberta-v3-large-lgbm-baseline-ad53b1-2.ipynb: This notebook implements the primary ensemble model, combining Deberta-v3-large embeddings with a LightGBM regressor, enriched by various NLP-driven features.

    threshold.ipynb: This notebook focuses on optimizing prediction thresholds for an ensemble model (LightGBM and XGBoost), and demonstrates a post-processing step often crucial for improving the QWK score in ordinal regression problems.

0-818-deberta-v3-large-lgbm-baseline-ad53b1-2.ipynb - Main Ensemble Model

This notebook serves as the core of the essay scoring solution. It combines the power of a large pre-trained language model with traditional machine learning techniques and rich handcrafted features.
Key Features and Techniques:

    Deberta-v3-large Integration:

        Utilizes pre-trained Deberta-v3-large models to generate essay embeddings.

        These embeddings are then incorporated as additional features for the LightGBM model.

        A 5-fold StratifiedKFold cross-validation strategy is applied to Deberta for generating Out-Of-Fold (OOF) predictions, which are crucial for stacking.

    Extensive Feature Engineering:

        Paragraph-level Features: Extracts metrics such as paragraph length, sentence count, word count, and spelling errors (using spacy and a custom vocabulary). Aggregations (max, mean, min, sum, kurtosis, quantiles) are computed per essay.

        Sentence-level Features: Processes sentences (segmented by periods) to derive sentence length and word counts, with similar aggregations.

        Word-level Features: Analyzes individual words for length and frequency.

        Text Vectorization: Employs TfidfVectorizer (for trigrams up to 6-grams) and CountVectorizer (for bigrams up to trigrams) to capture textual patterns and n-gram frequencies.

    LightGBM Regressor (LGBM):

        A powerful gradient boosting framework is used as the main regressor.

        It's trained with a custom objective function (qwk_obj) and evaluation metric (quadratic_weighted_kappa) designed to directly optimize the Quadratic Weighted Kappa (QWK) score, which is the competition's evaluation metric.

        A 15-fold StratifiedKFold cross-validation setup is used for robust model training and OOF prediction generation.

        Feature importance from LGBM is used for feature selection, retaining highly relevant features (e.g., top 13000 features).

    Ensemble Strategy:

        The Deberta OOF predictions are added as features to the LightGBM model, creating a stacking-like ensemble.

        The final predictions are derived by averaging the predictions from the 15 LightGBM folds.

Dependencies:

    pandas, numpy, polars (for efficient data handling)

    spacy (for NLP tasks like lemmatization and tokenization)

    transformers, datasets, torch (for Deberta model handling)

    lightgbm, scikit-learn (for LGBM, vectorizers, and metrics)

    scipy (for softmax)

    joblib (for loading pre-computed OOF predictions)

threshold.ipynb - Threshold Optimization and Ensemble

This notebook explores a crucial post-processing step for ordinal regression tasks: optimizing the classification thresholds. While the main model predicts continuous scores, the final submission requires integer scores (typically 1-6).
Key Features and Techniques:

    Threshold Optimization:

        Implements find_thresholds, a function that iteratively adjusts discrete classification thresholds to maximize the Quadratic Weighted Kappa (QWK) score. This is done by testing small perturbations around initial thresholds (e.g., [1.5, 2.5, 3.5, 4.5, 5.5]).

        The notebook applies this optimization to the Out-Of-Fold (OOF) predictions of its own ensemble, ensuring the thresholds are generalized to unseen data.

    Ensemble with XGBoost:

        Similar to the main notebook, it trains LightGBM models.

        It also integrates XGBRegressor into the ensemble, providing diversity.

        Both LightGBM and XGBoost models are trained with the custom qwk_obj and quadratic_weighted_kappa functions.

        A Predictor class is defined to combine the predictions of LGBM and XGBoost, with a weighted average (e.g., 0.749*light_preds + (1-0.749)*xgb_preds).

    Post-processing and Submission:

        The raw continuous predictions from the ensemble are converted to discrete scores using the optimized thresholds (e.g., pd.cut function).

        The final output is a submission.csv file.

    Overall Ensemble (Demonstration):

        The final cell in this notebook demonstrates how its submission.csv might be combined with another submission file (e.g., submission_svr_0806.csv, likely from a Support Vector Regressor model) by applying a weighted average on the final scores. This indicates a multi-level ensemble strategy.

Dependencies:

    pandas, numpy

    lightgbm, xgboost, scikit-learn

    aes2_preproces_cache_vectorize (likely a script or module derived from the feature engineering steps of the main notebook, containing shared preprocessing logic and feature selection).

    optuna (if hyperparameter tuning is enabled).

Usage

These notebooks are designed to be run in a Kaggle-like environment where input datasets are readily available.

    Data Setup: Ensure you have access to the learning-agency-lab-automated-essay-scoring-2 competition data and any auxiliary datasets (e.g., pre-computed embeddings, word lists) specified within the notebooks (e.g., /kaggle/input/...).

    Execution Order:

        It's recommended to run 0-818-deberta-v3-large-lgbm-baseline-ad53b1-2.ipynb first, as its Deberta predictions are used as features in the threshold.ipynb notebook (or at least conceptually, if not directly passed via intermediate files).

        Then, threshold.ipynb can be executed to perform threshold optimization and generate its part of the ensemble.

    Running Notebooks: Open the .ipynb files in a Jupyter environment and run all cells sequentially.

The notebooks will generate submission.csv files, which can then be submitted to the Kaggle competition or further combined with other models for a stronger ensemble.
