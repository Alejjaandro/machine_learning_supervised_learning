# Importar las bibliotecas necesarias
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("DataSet_Titanic.csv")

# Save the column "Sobreviviente" as  "predictors"
X = df.drop("Survived", axis=1)

# Save the column "Sobreviviente" as  "data_to_predict"
Y = df["Survived"]

# Split data: x = predictors, y = data_to_predict
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Initialize the model
random_forest_model = RandomForestClassifier()

# Define the hyperparameters to tune
params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 20, 30, 40, 50],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "bootstrap": [True, False],
}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=random_forest_model, 
                           param_grid=params, 
                           cv=5,
                           scoring="accuracy",
                           n_jobs=-1)

# Training the model
print("Training the model...")
grid_search.fit(X_train, Y_train)

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

# Get the best model
best_model = grid_search.best_estimator_

# Making predictions
Y_pred = best_model.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(Y_test, Y_pred)
print(f"Accuracy of the model: {accuracy * 100:.2f}%")
print("Classification report:")
print(classification_report(Y_test, Y_pred))

# Prediction of a fictional passenger
# Class, Sex, Age, SiblingsSpouses, FatherSons
fictional_passenger = pd.DataFrame({
    "Class": [3],
    "Sex": [1],
    "Age": [80],
    "SibSp": [0],
    "ParCh": [0]
})
print("Fictional passenger:")
print(fictional_passenger)

# Fictional passenger prediction (Class, Genrer, Age, SiblingsSpouses, FatherSons)
prediccion_ficticial = best_model.predict(fictional_passenger)

if prediccion_ficticial == 1:
    print("Fictional passenger survived.")
else:
    print("Fictional passenger did not survive.")

# Graphic the importances of the variables
# Create variables x (importance) e y (columns)
importance = best_model.feature_importances_
columns = X.columns
sns.barplot(x=importance, y=columns)
plt.title("Importance of the variables")
plt.show()