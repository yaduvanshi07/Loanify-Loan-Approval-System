# Loanify — Loan Approval Prediction System

An end-to-end supervised machine learning pipeline that predicts whether a loan application will be approved, using applicant financial and demographic data. The project covers data cleaning, exploratory data analysis (EDA), feature engineering, and comparative evaluation of three classification models.

## Overview

Loanify treats loan approval as a binary classification problem (`Approved` / `Not Approved`) and walks through the full workflow of a typical credit-risk ML task:

- Handling missing values in both numerical and categorical features
- Exploratory analysis of applicant income, credit score, DTI ratio, and loan amount against approval outcome
- Encoding categorical variables (label encoding and one-hot encoding)
- Feature scaling
- Training and evaluating three classifiers: Logistic Regression, K-Nearest Neighbors, and Gaussian Naive Bayes

## Dataset

The notebook expects a CSV file named `loan_approval_data.csv` in the project root, containing 1,000 applicant records with the following columns:

| Column | Description |
|---|---|
| `Applicant_ID` | Unique identifier for the applicant |
| `Applicant_Income` | Primary applicant's income |
| `Coapplicant_Income` | Co-applicant's income |
| `Employment_Status` | Salaried / Self-employed |
| `Age` | Applicant's age |
| `Marital_Status` | Married / Single |
| `Dependents` | Number of dependents |
| `Credit_Score` | Applicant's credit score |
| `Existing_Loans` | Number of existing loans |
| `DTI_Ratio` | Debt-to-income ratio |
| `Savings` | Applicant's savings amount |
| `Collateral_Value` | Value of collateral offered |
| `Loan_Amount` | Requested loan amount |
| `Loan_Term` | Loan repayment term |
| `Loan_Purpose` | Personal / Car / Business / etc. |
| `Property_Area` | Urban / Semiurban / Rural |
| `Education_Level` | Graduate / Not Graduate |
| `Gender` | Applicant's gender |
| `Employer_Category` | Category of employer |
| `Loan_Approved` | Target variable — Yes / No |

> **Note:** The dataset file is not included in this repository. Place `loan_approval_data.csv` in the project root before running the notebook.

## Project Workflow

### 1. Data Cleaning
- Numerical columns imputed using mean strategy (`SimpleImputer`)
- Categorical columns imputed using most-frequent strategy

### 2. Exploratory Data Analysis
- Class distribution of `Loan_Approved` (pie chart)
- Gender distribution (bar plot)
- Applicant income distribution (histogram)
- Boxplots comparing `Applicant_Income`, `Loan_Amount`, `DTI_Ratio`, and `Credit_Score` against loan approval status

### 3. Encoding
- `Education_Level` and `Loan_Approved` encoded with `LabelEncoder`
- `Employment_Status`, `Marital_Status`, `Loan_Purpose`, `Property_Area`, `Gender`, and `Employer_Category` encoded with `OneHotEncoder` (first category dropped)
- `Applicant_ID` dropped as a non-predictive identifier

### 4. Train/Test Split & Scaling
- 80/20 train-test split (`random_state=42`)
- Features standardized using `StandardScaler`

### 5. Model Training & Evaluation
Three classifiers were trained and evaluated on Accuracy, Precision, Recall, F1 Score, and Confusion Matrix.

## Results

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.865 | 0.783 | 0.770 | 0.777 |
| K-Nearest Neighbors (k=5) | 0.760 | 0.627 | 0.525 | 0.571 |
| Gaussian Naive Bayes | 0.865 | 0.804 | 0.738 | 0.769 |

Logistic Regression and Gaussian Naive Bayes performed best overall, with comparable accuracy (86.5%). Logistic Regression achieved the highest recall, while Naive Bayes achieved the highest precision. KNN underperformed relative to the other two models across all metrics.

## Tech Stack

- **Python**
- **pandas** — data manipulation
- **seaborn / matplotlib** — visualization
- **scikit-learn** — preprocessing, modeling, and evaluation
  - `SimpleImputer`, `LabelEncoder`, `OneHotEncoder`, `StandardScaler`
  - `LogisticRegression`, `KNeighborsClassifier`, `GaussianNB`
  - `train_test_split`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`

## Getting Started

### Prerequisites
```bash
pip install pandas seaborn matplotlib scikit-learn jupyter
```

### Running the Notebook
1. Clone the repository
   ```bash
   git clone https://github.com/yaduvanshi07/Loanify-Loan-Approval-System.git
   cd Loanify-Loan-Approval-System
   ```
2. Add `loan_approval_data.csv` to the project root
3. Launch Jupyter and run `Credit.ipynb`
   ```bash
   jupyter notebook Credit.ipynb
   ```

## Repository Structure
```
Loanify-Loan-Approval-System/
├── Credit.ipynb    # Main notebook: EDA, preprocessing, and model training
├── .gitignore
└── README.md
```

## Possible Improvements
- Hyperparameter tuning (e.g., grid search for KNN's `n_neighbors`)
- Cross-validation instead of a single train-test split
- Additional models (Random Forest, XGBoost) for comparison
- Handling class imbalance if present in the target variable
- Feature importance / model interpretability analysis
- Persisting the trained model and CSV dataset for reproducibility
