# Bank Marketing Classification – Machine Learning Assignment 2

**Student Name:** Taru Sharma  
**BITS ID:** 2026aj05566  
**Programme:** M.Tech Artificial Intelligence and Machine Learning  
**Subject:** Machine Learning  
**Assignment:** Assignment 2  
**Institution:** BITS Pilani – Work Integrated Learning Programmes

---

## 1. Problem Statement

The aim of this assignment is to build machine learning classification models that can predict whether a customer will subscribe to a bank's term deposit.

The prediction is based on customer details, previous marketing campaign information and economic indicators. I have used the Bank Marketing dataset for this task and implemented five different classification algorithms.

The main objective is to train the models, compare their performance using different evaluation metrics and build a Streamlit application where the models can be tested using an uploaded CSV file.

The five models implemented are:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

## 2. Dataset Description

**Dataset:** Bank Marketing Dataset  
**Source:** UCI Machine Learning Repository  
**File used:** `bank-additional-full.csv`

The dataset contains information about direct marketing campaigns conducted by a Portuguese banking institution. The target variable indicates whether a customer subscribed to a term deposit.

The target column is `y`:
- `yes` – Customer subscribed to the term deposit.
- `no` – Customer did not subscribe.

### Dataset details

| Description | Details |
|---|---|
| Original records | 41,188 |
| Records after duplicate removal | 41,176 |
| Original columns | 21 |
| Original input features | 20 |
| Target column | y |
| Problem type | Binary Classification |
| Training data | 80% |
| Testing data | 20% |

### Data preprocessing

Before training the models, I performed the following preprocessing steps:

- Checked the dataset for missing values and duplicate records.
- Removed 12 duplicate records.
- Dropped the `duration` column because it is only known after the call and could cause target leakage.
- Created a new feature called `previously_contacted` using the `pdays` column.
- Replaced the special value 999 in `pdays` with a missing value so it could be handled through imputation.
- Used median imputation and StandardScaler for numerical features.
- Used most-frequent imputation and OneHotEncoder for categorical features.
- Converted the target values into binary format (no = 0, yes = 1).

After preprocessing and feature engineering, the models used 20 input features.

The dataset was divided into training and testing sets using an 80:20 split. I used stratified sampling and `random_state=42` to maintain the class distribution in both sets.

The preprocessing steps were fitted on the training data and then applied to the test data.

## 3. GitHub Repository Link

**Repository:** https://github.com/Tarusharmabits2026aj05566/bank-marketing-classifier-2026aj05566

The repository contains the notebook, Streamlit application, test dataset, model comparison results, saved trained models and preprocessing artifact.

### Project structure

```text
bank-marketing-classifier-2026aj05566/
│
├── app.py
├── requirements.txt
├── README.md
├── ML_Assignment_2_Bank_Marketing.ipynb
├── test_data.csv
├── model_comparison.csv
│
└── model/
    ├── preprocessor.joblib
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## 4. Models Used and Performance Comparison

All five models were trained using the same training dataset and evaluated on the same test dataset.

The following metrics were calculated for each model:

- Accuracy
- AUC Score
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### 4.1 Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8981 | 0.8005 | 0.6469 | 0.2112 | 0.3184 | 0.3302 |
| Decision Tree | 0.8418 | 0.6312 | 0.3171 | 0.3502 | 0.3328 | 0.2437 |
| KNN | 0.8951 | 0.7403 | 0.5653 | 0.2985 | 0.3907 | 0.3600 |
| Gaussian Naive Bayes | 0.8230 | 0.7761 | 0.3356 | 0.5830 | 0.4260 | 0.3478 |
| Support Vector Machine | 0.8993 | 0.7971 | 0.6800 | 0.2015 | 0.3109 | 0.3335 |

### 4.2 Observations on Each Model

**1. Logistic Regression**

Logistic Regression achieved an accuracy of 89.81% and the highest AUC score of 0.8005. Its precision is reasonably good, but recall is low at 0.2112. This means that although its predictions are often correct when it identifies a subscriber, it misses a considerable number of actual subscribers.

**2. Decision Tree Classifier**

The Decision Tree achieved an accuracy of 84.18%. Its AUC score was 0.6312, which was the lowest among the five models. It achieved a recall of 0.3502, but its overall performance was lower than the other models in this experiment.

**3. K-Nearest Neighbors (KNN)**

KNN achieved an accuracy of 89.51% and an F1 score of 0.3907. It also obtained the highest MCC score of 0.3600. Compared with Logistic Regression and SVM, KNN provided a better balance between precision and recall.

**4. Gaussian Naive Bayes**

Gaussian Naive Bayes achieved an accuracy of 82.30%. It recorded the highest recall of 0.5830 and the highest F1 score of 0.4260. Its precision was lower than the other models, but it was able to identify more actual subscribers. This makes it useful when finding potential subscribers is the main objective.

**5. Support Vector Machine (SVM)**

The Linear Support Vector Machine achieved the highest accuracy of 89.93% and the highest precision of 0.6800. Its AUC score was 0.7971. However, its recall was only 0.2015, which means it missed many actual subscribers.

## 4.3 Overall Winner

Based on the results, I selected **Gaussian Naive Bayes** as the overall winner for this dataset.

It achieved the highest F1 score of 0.4260 and the highest recall of 0.5830. Since the objective is to identify customers who are likely to subscribe to a term deposit, identifying more actual subscribers is important.

However, the best model can change depending on the metric being considered:

| Metric | Best Performing Model |
|---|---|
| Accuracy | Support Vector Machine |
| AUC | Logistic Regression |
| Precision | Support Vector Machine |
| Recall | Gaussian Naive Bayes |
| F1 Score | Gaussian Naive Bayes |
| MCC | KNN |

Therefore, Gaussian Naive Bayes is my preferred model for this particular classification objective, although it is not the best in every metric.

## 5. Streamlit Application

**Live Application:** https://bank-marketing-classifier-2026aj05566-cbeps3vxrzhqtxf4iz6if5.streamlit.app/
I developed a Streamlit application to make the model evaluation interactive.

The application provides the following features:

- Upload a test dataset in CSV format.
- Select any one of the five trained classification models.
- View Accuracy, AUC, Precision, Recall, F1 Score and MCC.
- View the confusion matrix.
- View the classification report.

The application loads the saved preprocessing artifact and trained models from the repository. It does not train the models again when a test CSV is uploaded.

The uploaded CSV must contain all the required input features along with the target column `y`.

## 6. Conclusion

Through this assignment, I implemented five classification algorithms on the Bank Marketing dataset and compared their performance using six different evaluation metrics.

One important observation was that accuracy alone does not give a complete picture of model performance, especially when the dataset has an imbalanced target variable.

Although SVM achieved the highest accuracy, Gaussian Naive Bayes performed better in terms of recall and F1 score. For this reason, I selected Gaussian Naive Bayes as the preferred model for identifying potential term deposit subscribers.

Finally, I developed a Streamlit application that allows users to upload test data and evaluate the saved models through an interactive interface.
