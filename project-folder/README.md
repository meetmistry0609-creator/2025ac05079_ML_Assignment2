# Dry Bean Classification - ML Assignment 2

## a. Problem Statement

Dry beans are classified and sold as different registered varieties. Traditionally, identifying the variety of a bean may involve manual visual inspection, which can be time-consuming and difficult to perform consistently at scale.

The objective of this project is to develop an automated **multi-class classification system** that identifies the variety of a dry bean using physical and morphological characteristics extracted from images of the beans.

Five classical machine learning classification models are implemented, trained, and compared on the same dataset:

1. Logistic Regression
2. Decision Tree
3. k-Nearest Neighbors (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

The models are evaluated using the following performance metrics:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

An interactive **Streamlit web application** is also developed and deployed. The application allows users to upload test data, select individual models or compare all models, view evaluation metrics, and examine confusion matrices and classification reports.

---

## b. Dataset Description

The **Dry Bean Dataset** from the UCI Machine Learning Repository is used for this project.

**Dataset:** [UCI Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset)

### Dataset Characteristics

| Property | Description |
|---|---|
| Dataset | Dry Bean Dataset |
| Source | UCI Machine Learning Repository |
| Dataset ID | 602 |
| Original Instances | 13,611 |
| Instances after duplicate removal | 13,543 |
| Duplicate Rows Removed | 68 |
| Base Features | 16 numeric features |
| Engineered Features | 3 additional features |
| Total Features Used | 19 |
| Target Variable | `Class` |
| Number of Classes | 7 |
| Problem Type | Multi-class Classification |

The original dataset contains **13,611 instances**. During preprocessing, **68 exact duplicate rows were removed**, resulting in **13,543 unique observations**.

### Input Features

The original dataset contains 16 numerical features describing the physical and morphological characteristics of each dry bean.

These include:

- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- AspectRatio
- Eccentricity
- Roundness
- Compactness
- ShapeFactor1
- ShapeFactor2
- ShapeFactor3
- ShapeFactor4

Three additional engineered ratio-based features were created:

- ConvexityDeficit
- AreaPerimeterRatio
- DiameterRatio

Therefore, the final dataset contains **19 features**, satisfying the feature requirement for the assignment.

### Target Classes

The target variable is `Class`, representing seven registered varieties of dry beans:

1. Barbunya
2. Bombay
3. Cali
4. Dermason
5. Horoz
6. Seker
7. Sira

This is therefore a **7-class multi-class classification problem**.

### Class Distribution

The dataset is moderately imbalanced. Dermason is the majority class with approximately 3,500 observations, while Bombay is the minority class with approximately 520 observations.

To handle the class distribution appropriately:

- A **stratified train-test split** was used.
- **Weighted-average Precision, Recall, and F1 Score** were used for evaluation.

---

## d. Models Used

Five machine learning classification models were implemented and evaluated using the same Dry Bean dataset.

All five models use an identical **80:20 stratified train-test split**. Each model is implemented using a `StandardScaler --> Classifier` pipeline.

### 1. Logistic Regression
### 2. Decision Tree
### 3. k-Nearest Neighbors (kNN)
### 4. Gaussian Naive Bayes
### 5. Random Forest (Ensemble)

---

## e. Model Performance Comparison

The performance of all five models was evaluated using:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | **0.9188** | **0.9935** | **0.9193** | **0.9188** | **0.9189** | **0.9019** |
| Decision Tree | 0.9048 | 0.9644 | 0.9052 | 0.9048 | 0.9049 | 0.8848 |
| kNN | 0.9144 | 0.9851 | 0.9151 | 0.9144 | 0.9144 | 0.8964 |
| Naive Bayes | 0.8952 | 0.9898 | 0.8979 | 0.8952 | 0.8953 | 0.8740 |
| Random Forest (Ensemble) | 0.9169 | 0.9926 | 0.9170 | 0.9169 | 0.9169 | 0.8995 |

---

## f. Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Logistic Regression provides the best overall performance among the evaluated models. It achieves the highest AUC of **0.9935** and the highest MCC of **0.9019**. The results indicate that the seven bean varieties are reasonably well separated using the available shape and size features. It is also computationally efficient and fast to train and predict. |
| **Decision Tree** | Decision Tree achieves an accuracy of **0.9048** and an MCC of **0.8848**, making it the weakest tree-based model in this comparison. A single decision tree can be sensitive to noisy feature splits, whereas ensemble methods can reduce this effect. |
| **kNN** | kNN achieves **0.9144 accuracy** and an AUC of **0.9851**. After feature standardization, distance-based similarity provides a reasonable representation of the physical similarity between bean samples. However, prediction can be slower because the algorithm compares new observations with training observations. |
| **Naive Bayes** | Naive Bayes achieves **0.8952 accuracy** and **0.8740 MCC**, which are the lowest among the five models. The relatively high correlation between several size-related features may not align well with the conditional-independence assumption of Naive Bayes. However, it still achieves a strong AUC of **0.9898**. |
| **Random Forest (Ensemble)** | Random Forest provides strong and balanced performance, achieving **0.9169 accuracy**, **0.9926 AUC**, and **0.8995 MCC**. Its ensemble of multiple decision trees helps reduce the overfitting associated with an individual decision tree. |

---

## g. Overall Winner for the Dataset

### Logistic Regression

**Logistic Regression is the overall winner for this Dry Bean dataset.**

It achieves the highest:

- **AUC:** 0.9935
- **MCC:** 0.9019

It also provides competitive Accuracy, Precision, Recall, and F1 Score.

Although Random Forest is a more complex ensemble model, Logistic Regression performs slightly better on this particular dataset. This indicates that the Dry Bean classes are sufficiently separable using the available numerical shape and size features.

Random Forest is a very close second, with an MCC of **0.8995** compared with Logistic Regression's **0.9019**.

The difference in MCC is approximately:

**0.9019 − 0.8995 = 0.0024**

Therefore, Logistic Regression is selected as the best-performing model for this dataset based on the overall evaluation.

---

## h. Additional Model Performance Observations

Inspection of the confusion matrices shows that most of the classification errors occur between bean varieties that have similar physical and morphological characteristics.

The major areas of confusion are:

- **Sira ↔ Dermason**
- **Barbunya ↔ Cali**

These classes have relatively similar size and shape characteristics, making them more difficult to distinguish.

On the other hand, **Bombay** is visually and morphologically more distinct from the other classes and is therefore rarely confused with them.

---

## i. Streamlit Application

An interactive Streamlit web application was developed to provide a user-friendly interface for evaluating the trained machine learning models.

### Application Features

The application provides the following functionality:

- CSV test dataset upload
- Support for raw UCI feature columns and engineered feature columns
- Machine learning model selection
- Option to compare all five models
- Accuracy display
- AUC display
- Precision display
- Recall display
- F1 Score display
- MCC display
- Confusion matrix visualization
- Classification report
- Model-wise performance comparison
- Prediction analysis

### Dataset Upload

Users can upload their own test CSV containing:

- The 16 original UCI Dry Bean features, or
- The 19 features including the engineered features

The application can also use the bundled sample test dataset.

### Model Selection

The application provides a model-selection dropdown containing:

- Logistic Regression
- Decision Tree
- kNN
- Naive Bayes
- Random Forest

An option to **Compare All Models** is also provided.

### Evaluation Metrics

When the uploaded dataset contains the ground-truth `Class` column, the application calculates and displays:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- MCC

### Confusion Matrix and Classification Report

The application displays the confusion matrix and classification report for the selected model.

When all models are compared, the confusion matrix can also be viewed for the best-performing model based on MCC.

---

## k. Project Files

The GitHub repository contains the following files:

| File | Description |
|---|---|
| `app.py` | Streamlit web application |
| `README.md` | Project documentation |
| `requirements.txt` | Python libraries required to run the project |
| `test_data.csv` | Test dataset used for model evaluation |
| `ML-Assignment2.ipynb` | Complete machine learning implementation and evaluation notebook |
| `logistic_regression.pkl` | Trained Logistic Regression model |
| `decision_tree.pkl` | Trained Decision Tree model |
| `knn.pkl` | Trained kNN model |
| `naive_bayes.pkl` | Trained Gaussian Naive Bayes model |
| `random_forest.pkl` | Trained Random Forest model |
| `scaler.pkl` | StandardScaler used in the model pipelines |

---

## l. Model Evaluation Methodology

The following workflow was followed to develop and evaluate the classification models:

1. Load the Dry Bean dataset.
2. Inspect the dataset and its class distribution.
3. Identify and remove exact duplicate rows.
4. Separate input features and the target variable.
5. Create three engineered ratio-based features.
6. Prepare the final 19-feature dataset.
7. Encode the target classes.
8. Perform an 80:20 stratified train-test split.
9. Apply StandardScaler to the input features.
10. Train Logistic Regression.
11. Train Decision Tree.
12. Train kNN.
13. Train Gaussian Naive Bayes.
14. Train Random Forest.
15. Generate predictions and class probabilities.
16. Calculate Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
17. Generate confusion matrices and classification reports.
18. Compare the performance of all five models.
19. Save the trained models.
20. Integrate the models into the Streamlit application.
21. Deploy the application using Streamlit Community Cloud.

---

## m. Technologies Used

The following technologies and libraries were used:

- **Python**
- **Jupyter Notebook**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **Matplotlib**
- **Seaborn**
- **Streamlit**
- **Joblib**

