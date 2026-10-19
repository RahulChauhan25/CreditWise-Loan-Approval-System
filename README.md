# 💳 CreditWise Loan Approval Prediction System

A machine learning classification project that predicts whether a loan application is likely to be approved or rejected based on an applicant's financial profile, credit history, employment status, and other relevant attributes.

## 📌 Problem Statement

SecureTrust Bank receives loan applications from customers across urban and rural regions. Its traditional manual verification process requires loan officers to examine income, employment details, credit scores, existing loans, and other financial information.

This process can be time-consuming and inconsistent, potentially leading to good applicants being rejected and high-risk applicants being approved.

The **CreditWise Loan System** explores how machine learning can analyse historical loan application data and predict loan approval outcomes to support faster, more consistent, and data-driven lending decisions.

## 🎯 Project Objectives

- Analyse applicant demographics, financial information, and credit-related attributes.
- Perform exploratory data analysis (EDA) to understand patterns in loan approval.
- Handle missing values and prepare data for machine learning.
- Encode categorical variables and scale numerical features.
- Explore relationships between input features and loan approval outcomes.
- Train and evaluate multiple classification algorithms.
- Compare models using precision, recall, F1-score, accuracy, and confusion matrices.
- Experiment with feature engineering to explore potential improvements in predictive performance.

## 📊 Dataset Description

The dataset, `loan_approval_data.csv`, contains **1,000 loan application records and 20 columns**, including the target variable.

| Category | Features |
|---|---|
| Applicant Information | Applicant_ID, Age, Gender, Marital_Status, Dependents |
| Income & Employment | Applicant_Income, Coapplicant_Income, Employment_Status, Employer_Category |
| Credit Profile | Credit_Score, Existing_Loans, DTI_Ratio |
| Financial Assets | Savings, Collateral_Value |
| Loan Details | Loan_Amount, Loan_Term, Loan_Purpose |
| Other Information | Property_Area, Education_Level |
| Target Variable | Loan_Approved |

### Target Variable

- `1` — Loan approved
- `0` — Loan rejected

## 🛠️ Tech Stack

- **Programming Language:** Python
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Development Environment:** Jupyter Notebook

## ⚙️ Project Workflow

### 1. Data Loading and Inspection
- Loaded the loan application dataset using Pandas.
- Examined dataset dimensions, column information, descriptive statistics, and missing values.
- Inspected the structure and distribution of the target variable.

### 2. Data Cleaning and Preprocessing
- Identified missing values in numerical and categorical columns.
- Applied mean imputation to numerical features.
- Applied most-frequent-value imputation to categorical features.
- Removed the applicant ID column before model training.

### 3. Exploratory Data Analysis (EDA)
Performed visual and statistical analysis to investigate:
- Loan approval distribution.
- Education level and gender distributions.
- Applicant and co-applicant income distributions.
- Relationships between loan approval and applicant income.
- Credit scores, debt-to-income ratio (DTI), and savings across approval outcomes.

### 4. Categorical Encoding
Prepared categorical variables for machine learning using:
- **Label Encoding:** Applied to `Education_Level` and `Loan_Approved`.
- **One-Hot Encoding:** Applied to employment status, marital status, loan purpose, property area, gender, and employer category.

### 5. Correlation Analysis
- Calculated correlations between numerical features.
- Examined the relationship between input variables and the loan approval target.
- Visualized the correlation matrix using a heatmap.

### 6. Train-Test Split and Feature Scaling
- Split the dataset into training and testing sets using an 80:20 ratio.
- Used `random_state=42` for reproducibility.
- Standardized numerical model inputs using `StandardScaler`.

### 7. Model Training and Evaluation

Implemented and evaluated the following classification algorithms:

**Logistic Regression**
- Used as a baseline classification model for predicting loan approval.

**K-Nearest Neighbors (KNN)**
- Implemented KNN with `n_neighbors=5` to classify applications based on neighbouring observations.

**Gaussian Naive Bayes**
- Applied a probabilistic classification approach to predict loan approval outcomes.

Model evaluation includes:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

The notebook identifies Gaussian Naive Bayes as the preferred model based on precision in its initial comparison. Actual performance should be verified from the notebook's evaluation output.

### 8. Feature Engineering
Experimented with additional features to capture potential nonlinear relationships:
- `DTI_Ratio_sq` — squared debt-to-income ratio.
- `Credit_Score_sq` — squared credit score.

Retrained and evaluated the classification models using the engineered feature set to explore potential performance improvements.

## 📁 Repository Structure

```text
CreditWise-Loan-System/
│
├── credit_wise.ipynb
├── loan_approval_data.csv
└── README.md
```

## 🚀 Getting Started

### Prerequisites

Install Python 3 and the required libraries.

### Installation

**1. Clone the repository**

Replace the URL below with your actual GitHub repository URL.

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
```

**2. Navigate to the project directory**

```bash
cd <YOUR_REPOSITORY_FOLDER>
```

**3. Install the required libraries**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

**4. Launch Jupyter Notebook**

```bash
jupyter notebook
```

**5. Run the project**

Open `credit_wise.ipynb` and execute the notebook cells in order. Keep `loan_approval_data.csv` in the same directory as the notebook.

## 💡 Potential Business Applications

- Supporting preliminary loan application screening.
- Identifying patterns associated with historical approval decisions.
- Helping financial analysts compare applicant profiles.
- Providing insights into the relationships between credit scores, income, and loan outcomes.
- Supporting the development of data-driven lending workflows.

These are potential applications rather than validated business outcomes. The model is an educational project and should not be used to make real lending decisions without further validation.

## 🔮 Future Improvements

- Compare additional classification algorithms such as Random Forest, XGBoost, and Support Vector Machines.
- Tune hyperparameters using cross-validation.
- Address class imbalance if present.
- Improve the preprocessing workflow using Scikit-learn pipelines.
- Evaluate model calibration and decision thresholds.
- Add explainability using SHAP or feature importance techniques.
- Develop an interactive application for exploring loan predictions.
- Assess fairness, bias, and performance across applicant groups before any real-world deployment.

---

