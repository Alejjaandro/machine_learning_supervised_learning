import os
import textwrap

from pathlib import Path
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

def tree_predictor(passenger):
    # Import data
    df = pd.read_csv(Path(__file__).parent / "DataSet_Titanic.csv")
    
    # Divide data into X and Y. X is the data we will use to predict Y
    X = df.drop("Survived", axis=1)
    Y = df["Survived"]

    # Divide data into validation and training data
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Create base model
    base_model = DecisionTreeClassifier(random_state=42)

    # Define parameters for GridSearchCV 
    parameters = {
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "criterion": ["gini", "entropy"]
    }

    # Implement GridSearchCV
    grid_search = GridSearchCV(estimator=base_model, param_grid=parameters, cv=5, scoring='accuracy')

    # Training GridSearchCV with our data and searching for the best parameters.
    print("\nTraining GridSearchCV...")
    grid_search.fit(X_train, Y_train)
    grid_model = grid_search.best_estimator_

    # Train our model
    print("Training model...\n")
    grid_model.fit(X_train, Y_train)
    
    # Make predictions over our data
    prediction_grid_model = grid_model.predict(X_valid)

    # Calculate the accuracy of the model
    print(f"Accuracy of the GridSearch model: {round(accuracy_score(Y_valid, prediction_grid_model)*100, 2)}%\n")
                    
    # Fictional passenger prediction (Class, Genrer, Age, SiblingsSpouses, FatherSons)
    prediccion_ficticial = grid_model.predict(passenger)
    surviving_prob = round(grid_model.predict_proba(passenger)[0][1] * 100, 2)
    
    # SHAP values for each variable of the passenger
    explainer = shap.Explainer(grid_model, X_train)
    
    shap_values = explainer(passenger)
    
    shap_single_value = shap_values[0][:, 1]
    increase_survival = np.round(shap_single_value.values * 100, 2)
    base_survival = round(shap_single_value.base_values * 100, 2)
    
    total_survival_inceased = round(increase_survival.sum(), 2)
    
    description = textwrap.dedent(f'''
    Your had a base survival probability of {base_survival.round(2)}%.
    
    Each variable increases or decreases the survival probability of the passenger:
    - "Class" increases the survival probability by {increase_survival[0]}%.
    - "Sex" increases the survival probability by {increase_survival[1]}%.
    - "Age" increases the survival probability by {increase_survival[2]}%.
    - "SibSp" increases the survival probability by {increase_survival[3]}%.
    - "ParCh" increases the survival probability by {increase_survival[4]}%.
    For a total increase of {total_survival_inceased}%.
    
    Finally, the survival probability of the passenger is {round(base_survival + total_survival_inceased, 2)}%.
    ''')
    
    result = {
        "result": f"\nYour passenger has a {surviving_prob}% probability of surviving!\n",
        "description": description
    }
    
    return result

# Test the model with a fictional passenger
passenger = pd.DataFrame({
    "Class": [2],
    "Sex": [1],
    "Age": [24],
    "SibSp": [0],
    "ParCh": [0]
})

# output = tree_predictor(passenger)
# print(output["result"])
