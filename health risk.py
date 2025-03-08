import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Create a small synthetic dataset
np.random.seed(42)
data = pd.DataFrame({
    'bmi': np.random.uniform(15, 40, 100),
    'age': np.random.randint(18, 70, 100),
    'blood_pressure': np.random.uniform(90, 180, 100)
})

data['Health_Risk'] = (data['bmi'] > 30).astype(int)  # 1 for Obese, 0 otherwise

# Define features and target variables
X = data[['bmi', 'age', 'blood_pressure']]
y = data['Health_Risk']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train a RandomForest model with limited complexity to reduce overfitting
model = RandomForestClassifier(n_estimators=20, max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Risk', 'Risk'], yticklabels=['No Risk', 'Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Classification report
print(classification_report(y_test, y_pred))

# Plot feature importance
feature_importances = model.feature_importances_
plt.figure(figsize=(10, 5))
sns.barplot(x=X.columns, y=feature_importances)
plt.xticks(rotation=45)
plt.title('Feature Importances')
plt.show()

# Function to provide health recommendations
def health_recommendation(bmi, age):
    if bmi < 18.5:
        return "Increase calorie intake and consume nutrient-dense foods."
    elif 18.5 <= bmi < 25:
        return "Maintain a balanced diet and stay active."
    elif 25 <= bmi < 30:
        return "Monitor your diet, reduce processed foods, and exercise regularly."
    else:
        return "Consult a healthcare provider, follow a structured diet, and engage in physical activity."

# Example usage
sample_bmi = 32
sample_age = 45
print(f'Recommendation: {health_recommendation(sample_bmi, sample_age)}')
