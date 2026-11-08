# Stroke Prediction: Classification Models with a Streamlit App

Machine Learning, Assignment 2 (M.Tech DSE, BITS Pilani WILP)

## a. Problem statement

Predict whether a patient is likely to have a **stroke** (`stroke` = 1) or not (`stroke` = 0) from basic demographic, lifestyle and health records. This is a **binary classification** problem. Five classifiers are implemented on the same dataset, evaluated on the same held-out test data with six metrics (Accuracy, AUC, Precision, Recall, F1, MCC), and demonstrated through an interactive Streamlit web application.

## b. Dataset description

| Item | Detail |
|---|---|
| Name | Stroke Prediction Dataset (`healthcare-dataset-stroke-data.csv`) |
| Source | Kaggle |
| Instances | 5,110 patient records |
| Columns | 12 in the file: `id`, 10 predictor features, and the target `stroke` |
| Model inputs | 10 raw features, which become 21 columns after one-hot encoding |
| Target | `stroke` (1 = stroke, 0 = no stroke) |
| Class balance | 249 stroke (4.9%) vs 4,861 no stroke (95.1%): heavily imbalanced |

**Features**

| Feature | Type | Description |
|---|---|---|
| gender | categorical | Male / Female / Other |
| age | numeric | Age in years |
| hypertension | binary | 1 if the patient has hypertension |
| heart_disease | binary | 1 if the patient has heart disease |
| ever_married | categorical | Yes / No |
| work_type | categorical | Private, Self-employed, Govt_job, children, Never_worked |
| Residence_type | categorical | Urban / Rural |
| avg_glucose_level | numeric | Average blood glucose level |
| bmi | numeric | Body mass index (201 values missing) |
| smoking_status | categorical | never smoked, formerly smoked, smokes, Unknown |

**Workflow**

1. **Train/test split:** 80% / 20%, stratified on `stroke`, `random_state=42`. Train = 4,088 rows (199 stroke), test = 1,022 rows (50 stroke). The test split is saved as `test_data.csv`; the training split as `model/train_data.csv`.
2. **Preprocessing** (fitted on the training data only, then applied to the test data): `id` dropped; missing `bmi` filled with the training median; numeric columns standardised; categorical columns one-hot encoded.
3. **Data skewness removal (sampling):** the minority (stroke) class in the **training split only** is randomly oversampled until both classes have 3,889 rows. The test split is never resampled, so reported metrics reflect the real class distribution.
4. **Model training and testing:** each model is a scikit-learn `Pipeline` (preprocessing + classifier), trained on the balanced training data and scored on the untouched test data.

## c. GitHub repository link

`https://github.com/2025ns04118/stroke-ml-app/tree/main`

Live Streamlit app: `https://stroke-ml-app-beiy6eae4tequkjmdt4pjk.streamlit.app/`

```
stroke-ml-app/
|-- stroke_ml_assignment.ipynb  step-by-step notebook (split, train, evaluate, save)
|-- app.py                  Streamlit application (tabs: performance, curves, leaderboard, patient what-if, about)
|-- .streamlit/config.toml  app colour theme
|-- requirements.txt        Python dependencies
|-- README.md               this file
|-- test_data.csv           20% test split (upload this in the app)
|-- data/
|   |-- healthcare-dataset-stroke-data.csv   original Kaggle file
|-- model/
    |-- model_utils.py      preprocessing, the 5 models, oversampling, metrics
    |-- train_models.py     trains, evaluates and saves all models
    |-- train_data.csv      80% training split
    |-- *.joblib            saved fitted pipelines (one per model)
    |-- metrics_summary.csv test-set metrics for all models
```

**Run locally**

1. `pip install -r requirements.txt`
2. Open `stroke_ml_assignment.ipynb` and run all cells top to bottom. This splits the data, trains the five models, prints the metrics and saves everything the app needs. (`python model/train_models.py` does the same from the command line.)
3. `streamlit run app.py`, then upload `test_data.csv` in the sidebar.

## d. Models used

All five models were trained on the same balanced training split and evaluated on the same 1,022-row test set (50 stroke cases).

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.7446 | 0.8444 | 0.1375 | 0.8000 | 0.2346 | 0.2590 |
| Decision Tree | 0.7221 | 0.7773 | 0.1201 | 0.7400 | 0.2067 | 0.2168 |
| kNN | 0.8483 | 0.5998 | 0.0732 | 0.1800 | 0.1040 | 0.0416 |
| Naive Bayes | 0.2299 | 0.7850 | 0.0587 | 0.9800 | 0.1107 | 0.0956 |
| Support Vector Machine | 0.7681 | 0.7815 | 0.1184 | 0.5800 | 0.1966 | 0.1808 |

Settings: Logistic Regression (`max_iter=1000`); Decision Tree (`criterion="entropy"`, `max_depth=5` to limit overfitting); kNN (`k=5`); Naive Bayes (Gaussian); SVM (RBF kernel). AUC is computed from predicted probabilities (SVM: from its decision function). Precision, Recall and F1 are for the positive (stroke) class.

For context, a model that always predicts "no stroke" would score 95.1% accuracy on this test set while catching zero strokes. Accuracy alone is therefore misleading here, and AUC, Recall, F1 and MCC are more informative.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best AUC (0.8444) and best Precision, F1 and MCC of the five models. It catches 40 of the 50 stroke cases (Recall 0.80). Its linear decision boundary generalises well on this data, and it is fast to train and easy to interpret through its coefficients. Accuracy (0.74) is lower than the majority-class baseline because balanced training makes the model flag many healthy patients as risky. |
| Decision Tree | Entropy-based tree limited to depth 5. It catches 37 of 50 strokes (Recall 0.74) with accuracy 0.72, but its AUC (0.7773) is lower than Logistic Regression because a shallow tree produces only a few distinct score levels. Its rules are the most readable of all the models, but a single tree is sensitive to the duplicated minority rows created by oversampling. |
| kNN | Highest Accuracy (0.8483) but the weakest model: Recall 0.18 (only 9 of 50 strokes found), AUC 0.5998 and MCC 0.0416, close to random. After encoding there are 21 features, many of them sparse one-hot columns, which dilutes distance-based neighbours; k=5 also leaves most predictions dominated by healthy neighbours. It shows clearly why accuracy should not be trusted on imbalanced data. |
| Naive Bayes | Highest Recall (0.98, 49 of 50 strokes found) but the lowest Accuracy (0.2299) and Precision (0.0587), because it labels roughly 82% of all patients as stroke. Its AUC (0.7850) shows that it ranks patients reasonably well, but its probabilities are badly calibrated at the default 0.5 threshold. The Gaussian assumption does not fit binary/one-hot features, and the independence assumption is violated by correlated factors such as age, hypertension, heart disease and marital status. |
| Support Vector Machine | RBF kernel gives Accuracy 0.7681, AUC 0.7815, Recall 0.58 and F1 0.1966. It captures non-linear structure but does not beat the linear Logistic Regression here, and it finds fewer strokes (29 of 50). It is also the slowest model to train and predict. |
| **Overall, Winner for your dataset?** | **Logistic Regression.** It has the highest AUC (0.8444), Precision (0.1375), F1 (0.2346) and MCC (0.2590) while still detecting 80% of stroke cases. Naive Bayes has higher Recall, but only by flagging most patients, which is not useful in practice. Caveats: Precision is low for every model (most flagged patients do not have a stroke), and the test set has only 50 positive cases, so the metrics are noisy. These models are suitable for screening or ranking risk, not for diagnosis. |
