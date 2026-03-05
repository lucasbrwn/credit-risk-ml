# Credit Risk ML (Python)

A machine learning project that predicts whether a borrower will experience serious delinquency within 2 years. Built to practice data cleaning, model training, evaluation, and interpreting results using basic classification models.

## Features

- Data preprocessing:
  - Handles missing values in the dataset
  - Clips extreme outliers for stability
- Train/test splitting:
  - Stratified sampling to preserve class distribution
- Machine learning models:
  - Logistic Regression
  - Decision Tree
  - K-Nearest Neighbors (KNN)
- Model evaluation:
  - Classification report (precision, recall, F1-score)
  - Confusion matrix
- Business impact analysis:
  - Estimates financial impact of false positives and false negatives

## Models Used

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)

## Install and Run

From the project root:

```bash
pip install -r requirements.txt
python src/creditml.py
```
## Findings

- Logistic Regression
  - Performed well as a baseline model
  - Did not improve our hypothetical business model

- Decision Tree
  - Detected default patterns more efficiently than Logistic Regression
  - May overfit the data in some cases

- K-Nearest Neighbors (KNN)
  - Did not perform well on this dataset
  - Does not work as well with imbalanced datasets

## Model Comparison

The three models were compared using classification reports and confusion matrices.

Overall observations:

- Logistic Regression provided baseline performance.
- Decision Trees captured stronger relationships in the data but may overfit.
- KNN struggled with the dataset due to class imbalance.