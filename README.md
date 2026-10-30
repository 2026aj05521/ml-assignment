# ML Assignment 2 - Breast Cancer Classification

## a. Problem Statement
Binary classification to predict whether a breast tumour is malignant (0) or benign (1) based on 30 numeric features from digitised images of FNA of breast masses. Five ML classifiers are trained, evaluated, and served through a Streamlit web application.

---

## b. Dataset Description
| Property | Value |
|---|---|
| Name | Wisconsin Breast Cancer Dataset |
| Source | sklearn.datasets / UCI ML Repository |
| Instances | 569 |
| Features | 30 numeric |
| Target | Binary: 0 Malignant (212), 1 Benign (357) |
| Train/Test Split | 80/20 stratified, random_state=42 |
| Preprocessing | StandardScaler |

---

## c. GitHub Repository Link

https://github.com/2026aj05521/ml-assignment


---

## d. Models and Evaluation Metrics

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9298 | 0.9868 | 0.9444 | 0.9444 | 0.9444 | 0.8492 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations
| ML Model Name | Observation |
|---|---|
| Logistic Regression | Best overall - highest accuracy 98.25% and AUC 0.9954. Linear boundary works well. Very low false-negative rate makes it clinically reliable. |
| Decision Tree | Lowest performer (91.23% accuracy). Axis-aligned boundaries overfit training data. Sensitive to individual feature thresholds. |
| KNN | Good performance (95.61% accuracy). High recall means fewer missed malignant cases. Depends on feature scaling and k. |
| Naive Bayes | Moderate accuracy (92.98%) but high AUC (0.9868). Independence assumption violated by correlated features. |
| Support Vector Machine | Tied with LR on all hard-threshold metrics. RBF kernel finds near-perfect margin in high-dimensional space. |

### Overall Winner
Logistic Regression - best AUC and tied best on all other metrics. Highly interpretable and clinically reliable.

---

## Streamlit App
https://YOUR_APP_NAME.streamlit.app
