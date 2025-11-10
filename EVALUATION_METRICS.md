# 📊 Model Evaluation Metrics

This document describes the evaluation metrics used to assess the performance of the sentiment analysis model.

## Metrics Implemented

### 1. **Accuracy Score**
- Measures the overall correctness of predictions
- Formula: `(True Positives + True Negatives) / Total Predictions`
- Range: 0 to 1 (higher is better)

### 2. **F1 Score**
- Harmonic mean of Precision and Recall
- Provides a balanced measure of model performance
- Formula: `2 * (Precision * Recall) / (Precision + Recall)`
- Range: 0 to 1 (higher is better)
- Calculated for:
  - **Weighted F1**: Overall F1 score across all classes
  - **Per-Class F1**: F1 score for Negative and Positive classes separately

### 3. **Confusion Matrix**
- Shows the breakdown of predictions vs actual labels
- Contains four values:
  - **True Negatives (TN)**: Correctly predicted negative reviews
  - **False Positives (FP)**: Incorrectly predicted as positive (Type I error)
  - **False Negatives (FN)**: Incorrectly predicted as negative (Type II error)
  - **True Positives (TP)**: Correctly predicted positive reviews

### 4. **Classification Report**
- Detailed metrics including:
  - **Precision**: Accuracy of positive predictions
  - **Recall**: Ability to find all positive samples
  - **F1 Score**: Harmonic mean of precision and recall
  - **Support**: Number of samples in each class

## How to Generate Metrics

Run the training script to generate all metrics:

```bash
python MRA.py
```

This will output:
1. Accuracy scores for all three Naive Bayes models
2. F1 scores (weighted and per-class)
3. Confusion matrices for all models
4. Detailed classification report for the best model (Bernoulli Naive Bayes)
5. A visualization of the confusion matrix saved as `confusion_matrix.png`

## Model Comparison

The script compares three Naive Bayes classifiers:
- **Gaussian Naive Bayes**
- **Multinomial Naive Bayes**
- **Bernoulli Naive Bayes** (Selected as best model)

## Output Files

After running `MRA.py`, you will get:
- `confusion_matrix.png`: Visualization of the confusion matrix for the best model
- Console output with all metrics displayed

## Interpretation

### Confusion Matrix Interpretation:
```
                Predicted
              Negative  Positive
Actual Negative   TN      FP
       Positive   FN      TP
```

### Good Model Characteristics:
- High True Positives and True Negatives
- Low False Positives and False Negatives
- Balanced F1 scores for both classes
- High overall accuracy and weighted F1 score

## Example Output

```
============================================================
ACCURACY SCORES
============================================================
Gaussian Naive Bayes Accuracy =  0.XXXX
Multinomial Naive Bayes Accuracy =  0.XXXX
Bernoulli Naive Bayes Accuracy =  0.XXXX

============================================================
F1 SCORES
============================================================
Gaussian Naive Bayes F1 Score =  0.XXXX
Multinomial Naive Bayes F1 Score =  0.XXXX
Bernoulli Naive Bayes F1 Score =  0.XXXX

============================================================
CONFUSION MATRICES
============================================================
Bernoulli Naive Bayes Confusion Matrix:
[[TN  FP]
 [FN  TP]]
```

## Notes

- The confusion matrix visualization is automatically saved as `confusion_matrix.png`
- All metrics are calculated on the test set (20% of the data)
- The model is trained on 80% of the data
- Random state is set to 9 for reproducibility

