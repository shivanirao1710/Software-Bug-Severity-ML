import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv("bug_severity_dataset.csv")

# Preprocessing
le_reported_by = LabelEncoder()
data['Reported_By'] = le_reported_by.fit_transform(data['Reported_By'])
data['Severity'] = data['Severity'].map({'High': 2, 'Medium': 1, 'Low': 0})

# Features and target
X = data.drop(columns=['Bug_ID', 'Severity', 'Module_Affected'])
y = data['Severity']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize classifiers
svm_model = SVC(kernel='linear', random_state=42)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
knn_model = KNeighborsClassifier(n_neighbors=5)

# Train the classifiers
svm_model.fit(X_train, y_train)
rf_model.fit(X_train, y_train)
knn_model.fit(X_train, y_train)

# Make predictions
y_pred_svm = svm_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_knn = knn_model.predict(X_test)

# Evaluate the classifiers using accuracy
accuracy_svm = accuracy_score(y_test, y_pred_svm)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
accuracy_knn = accuracy_score(y_test, y_pred_knn)

print(f"Accuracy of SVM: {accuracy_svm:.4f}")
print(f"Accuracy of Random Forest: {accuracy_rf:.4f}")
print(f"Accuracy of KNN: {accuracy_knn:.4f}")

# Print classification reports in the terminal
print("\nClassification Report for SVM:")
print(classification_report(y_test, y_pred_svm, target_names=["Low", "Medium", "High"]))

print("\nClassification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf, target_names=["Low", "Medium", "High"]))

print("\nClassification Report for KNN:")
print(classification_report(y_test, y_pred_knn, target_names=["Low", "Medium", "High"]))

# Create a figure with subplots (2 rows, 2 columns) for the accuracy comparison and confusion matrices
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Accuracy Comparison Plot (in the first subplot)
accuracies = [accuracy_svm, accuracy_rf, accuracy_knn]
models = ['SVM', 'Random Forest', 'KNN']

axs[0, 0].bar(models, accuracies, color=['#CE93D8', '#82B1FF', '#00BCD4'])
axs[0, 0].set_title('Accuracy Comparison of Classifiers', fontsize=14)
axs[0, 0].set_xlabel('Models', fontsize=12)
axs[0, 0].set_ylabel('Accuracy', fontsize=12)
axs[0, 0].set_ylim(0, 1)

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, model_name, ax):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Low", "Medium", "High"], yticklabels=["Low", "Medium", "High"], ax=ax)
    ax.set_title(f'Confusion Matrix - {model_name}', fontsize=12)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)

# Plot confusion matrices for each model in the remaining subplots
plot_confusion_matrix(y_test, y_pred_svm, 'SVM', axs[0, 1])
plot_confusion_matrix(y_test, y_pred_rf, 'Random Forest', axs[1, 0])
plot_confusion_matrix(y_test, y_pred_knn, 'KNN', axs[1, 1])

# Adjust layout for better spacing and smaller font sizes
plt.tight_layout(pad=3.0)

# Show the combined plot (Accuracy and Confusion Matrices)
plt.show()

# Print the accuracy scores for each classifier
print("Accuracy Scores:")
print(f"SVM: {accuracy_svm:.4f}")
print(f"Random Forest: {accuracy_rf:.4f}")
print(f"KNN: {accuracy_knn:.4f}")
