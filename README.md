# Software-Bug-Severity-ML

## Overview
This project aims to classify the severity of software bugs based on various features such as code changes, complexity, and bug report details. The severity classification includes three categories: **High**, **Medium**, and **Low**. The dataset is synthetically generated and used to train machine learning models to predict bug severity.

The project uses three classification models:
- **Support Vector Machine (SVM)**
- **Random Forest Classifier**
- **K-Nearest Neighbors (KNN)**

### Features
- Bug ID
- File Changes
- Lines Added
- Lines Removed
- Code Complexity
- Bug Report Length
- Reported By (Developer, Tester, User)
- Module Affected
- Previous Bugs in Module
- Time to Fix
- Severity (Target)

## Files in the Repository

### 1. **Data Generation Script (`generato.py`)**

This script generates a synthetic dataset containing 1000 records of software bugs, each with features like bug ID, file changes, lines added/removed, code complexity, and more. The dataset is saved as a CSV file (`bug_severity_dataset.csv`) for use in training machine learning models.

#### Key Functions:
- `random_string(length)`: Generates random strings used for `Module_Affected`.
- Randomly generates features like `Bug_ID`, `File_Changes`, `Lines_Added`, `Lines_Removed`, `Code_Complexity`, `Bug_Report_Length`, `Reported_By`, `Module_Affected`, and others.
- Saves the dataset to a CSV file.

### 2. **Bug Severity Prediction Script (`bug.py`)**

This script loads the synthetic dataset and applies various machine learning models to predict the severity of bugs. The models used are SVM, Random Forest, and KNN. It evaluates each model's accuracy and displays the confusion matrix and classification report.

#### Key Steps:
1. **Data Preprocessing**:
   - Label encoding of categorical features like `Reported_By`.
   - Map `Severity` values ("High", "Medium", "Low") to numeric labels.
2. **Model Training and Evaluation**:
   - Splits the dataset into training and test sets (80/20 split).
   - Standardizes the feature values using `StandardScaler`.
   - Trains the models (SVM, Random Forest, and KNN) and evaluates their performance.
3. **Results**:
   - Accuracy of each model.
   - Classification reports and confusion matrices for each model.
   - Accuracy comparison plot.

### 3. **bug_severity_dataset.csv**

This CSV file contains the synthetically generated data for bug severity detection. The columns in the dataset are:
- `Bug_ID`: Unique identifier for each bug.
- `File_Changes`: Number of file changes associated with the bug.
- `Lines_Added`: Number of lines added in the code.
- `Lines_Removed`: Number of lines removed in the code.
- `Code_Complexity`: A numeric value representing the complexity of the code.
- `Bug_Report_Length`: Length of the bug report.
- `Reported_By`: The person who reported the bug (Developer, Tester, User).
- `Module_Affected`: Random string identifier for the affected module.
- `Previous_Bugs_in_Module`: Number of previous bugs reported in the module.
- `Time_to_Fix`: Time taken to fix the bug.
- `Severity`: The target variable representing the severity of the bug (High, Medium, Low).

## Requirements

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`

You can install the required packages using `pip`:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
