from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import data
csv_path = Path(__file__).resolve().parents[0] / "DataSet_Titanic.csv"
df = pd.read_csv(csv_path)

# Split data: x = predictors, y = data_to_predict 
x = df.drop("Sobreviviente", axis=1)
y = df["Sobreviviente"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Create model
no_GridSearch_model = LogisticRegression(random_state=42)

# Train model
no_GridSearch_model.fit(x_train, y_train)

# Predict
y_pred = no_GridSearch_model.predict(x_test)

# Check accuracy
no_GridSearchCV_accuracy = accuracy_score(y_test, y_pred)
print(f"No GridSearchCV accuracy: {round(no_GridSearchCV_accuracy*100, 2)}%")

#  ========== Now we use GridSearchCV to find the best parameters ==========
parameters = [
    # Only valid for solver = 'liblinear'
    {
        'C': [0.01, 0.1, 1, 10, 100],  # Regularization parameter
        'penalty': ['l1', 'l2'],  # Type of regularization
        'solver': ['liblinear']  # Optimization algorithm
    },
    # Only valid for solver = 'saga' and penalty = 'l1' or 'l2
    {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['saga'],
    },
    # Only valid for solver = 'saga' and penalty = 'elasticnet. Add l1_ratio to balance the two types of regularization for elasticnet
    {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['elasticnet'],
        'solver': ['saga'],
        'l1_ratio': [0.1, 0.5, 0.9] # Only saga supports elasticnet
    }
]


# Base model for Logistic Regression
modelo_logistico = LogisticRegression(random_state=42, max_iter=10000)

# Implement GridSearchCV
grid_search = GridSearchCV(estimator=modelo_logistico, param_grid=parameters, cv=5, scoring='accuracy')
grid_search.fit(x_train, y_train)

# print(f"Best parameters: {grid_search.best_params_}")
# print(f"Better score: {round(grid_search.best_score_*100, 2)}%")

# Obtain the best model
GridSearch_model = grid_search.best_estimator_

# Make predictions using the best model
predictions = GridSearch_model.predict(x_test)

# Check accuracy
GridSearchCV_accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy of the GridSearchCV model: {round(GridSearchCV_accuracy*100, 2)}%")

# ========== Now we compare the accuracy of the models ==========
if no_GridSearchCV_accuracy > GridSearchCV_accuracy:
    print("\nThe base model is better than the GridSearchCV model")
    print(no_GridSearch_model)
else:
    print("\nThe GridSearchCV model is better than the base model")
    print(GridSearch_model)