# Machine Learning Assignment 2

## a. Problem statement
This assignment focuses on building and comparing multiple classification models on a real-world dataset, evaluating them using standard metrics, and deploying an interactive Streamlit dashboard that can load saved models and test data without retraining.

The goal is to:
- choose a public classification dataset,
- train and compare five classifiers,
- evaluate them using Accuracy, AUC, Precision, Recall, F1 Score, and MCC,
- save the trained models,
- build a Streamlit app for inference and metric display,
- deploy the app on Streamlit Community Cloud.

## b. Dataset description
Dataset used: Breast Cancer Wisconsin (Diagnostic) dataset

- Source: scikit-learn built-in dataset, originally from UCI / Wisconsin Diagnostic Breast Cancer data
- Problem type: Binary classification
- Number of instances: 569
- Number of features: 30 numeric features
- Target classes: 0 = malignant, 1 = benign
- Feature type: Real-valued measurements (mean radius, texture, smoothness, etc.)
- Reason for selection: the dataset is widely used for classification benchmarking and satisfies the assignment requirement of at least 12 features and 500 instances.
- The full dataset has 569 instances. An 80/20 stratified split uses 455 records to train the models and saves only the 114 held-out records in `test_data.csv` for evaluation.

## c. GitHub Repository Link
https://github.com/avighanwat/ML-Assignment

This project contains:
- app.py
- requirements.txt
- README.md
- test_data.csv
- train_models.py
- model/ directory with all saved classifiers

## d. Models used
Models implemented:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier
5. Support Vector Machine

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Very strong performance with high accuracy, AUC, and balanced precision/recall. Very suitable for this dataset. |
| Decision Tree | Performs reasonably well, but has noticeably lower AUC and MCC than the linear and KNN-based models. It is more prone to overfitting. |
| KNN | Good results, especially recall and F1, but slightly behind Logistic Regression and SVM in overall metrics. |
| Naive Bayes | Acceptable performance, but it is less consistent than Logistic Regression and SVM for this continuous-feature classification problem. |
| Support Vector Machine | Achieves equal-best accuracy and excellent AUC/precision/recall, making it competitive with Logistic Regression. |

Overall winner for this dataset: Logistic Regression and SVM are tied for the best overall performance. Logistic Regression is slightly better in AUC (0.9954 vs 0.9950), while both achieve the same overall classification quality metrics.

## Deployment and usage
To run locally:

```bash
pip install -r requirements.txt
python train_models.py
streamlit run app.py
```

The application allows:
- CSV upload of test data,
- evaluation on the uploaded test records without retraining,
- model selection dropdown,
- display of evaluation metrics,
- confusion matrix visualization,
- comparison of all five models on the uploaded dataset.

## Streamlit Community Cloud deployment
1. Push this project to GitHub.
2. Open https://streamlit.io/cloud
3. Sign in using GitHub.
4. Select "New app".
5. Choose the repository, branch, and app.py as the entry file.
6. Click Deploy.

## Project files
- app.py: interactive dashboard
- requirements.txt: package dependencies
- train_models.py: model training and evaluation pipeline
- test_data.csv: test dataset used for evaluation
- model/ : saved trained model files
- README.md: project documentation

## Submission checklist
- [ ] GitHub repo link works
- [ ] Streamlit app link opens correctly
- [ ] App loads without errors
- [ ] All required features implemented
- [ ] README.md updated and included in the submitted PDF
- [ ] Screenshot of the assignment execution on BITS Virtual Lab uploaded
