# Importar las bibliotecas necesarias
from pathlib import Path
import pandas as pd
import numpy as np
import shap
import math
import textwrap
import time

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def random_forest_predictor(passenger, window):
    df = pd.read_csv(Path(__file__).parent / "TitanicDataset.csv")
    df = df.drop(["PassengerId", "Name", "Ticket", "Fare", "Cabin", "Embarked"], axis=1)
    df["Sex"] = df["Sex"].apply(lambda x: 1 if x == "male" else 0)

    # Check for missing values.
    nan_count_per_column = df.isnull().sum()
    
    # Fill missing values in the "Age" column with the median of the group.    
    df['Age'] = df.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

    # Divide data into X and Y. X is the data we will use to predict Y
    X = df.drop("Survived", axis=1)
    Y = df["Survived"]

    # Divide data into validation and training data
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=89)

    # Initialize the model
    random_forest_model = RandomForestClassifier(random_state=89)

    # Define the hyperparameters to tune
    params = {
        "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None],
        "n_estimators": [50, 100, 150],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }

    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=random_forest_model, param_grid=params, cv=5, scoring="accuracy", n_jobs=-1)

    # Training the model
    print("\nTraining GridSearchCV...")
    window.update_idletasks()
    
    grid_search.fit(X_train, Y_train)

    # Get the best model
    grid_model = grid_search.best_estimator_

    # Making predictions
    prediction_grid_model = grid_model.predict(X_valid)

    # Evaluating the model
    print(f"Accuracy of the GridSearch model on validation set: {round(accuracy_score(Y_valid, prediction_grid_model)*100, 2)}%")
    time.sleep(1)
    window.update_idletasks()
    
    # Fictional passenger prediction (Class, Genrer, Age, SiblingsSpouses, FatherSons)
    print("\nCalculating survival of the fictional passenger...")
    time.sleep(1)
    window.update_idletasks()

    surviving_prob = round(grid_model.predict_proba(passenger)[0][1] * 100, 2)
    
    # SHAP values for each variable of the passenger
    explainer = shap.Explainer(grid_model, X_train)    
    shap_values = explainer(passenger)
    
    shap_single_value = shap_values[0][:, 1]
    
    # Calculate the increase in survival probability for each variable
    increase_survival = np.round(shap_single_value.values * 100, 2)
    base_survival = round(shap_single_value.base_values * 100, 2)
    total_survival_increased = round(increase_survival.sum(), 2)    
    
    description = textwrap.dedent(f'''
    Your passenger have a base survival probability of {base_survival.round(2)}%.
    
    Each variable increases or decreases the survival probability of the passenger:
    - "Class" changes the survival probability by {increase_survival[0]}%.
    - "Sex" changes the survival probability by {increase_survival[1]}%.
    - "Age" changes the survival probability by {increase_survival[2]}%.
    - "SibSp" changes the survival probability by {increase_survival[3]}%.
    - "ParCh" changes the survival probability by {increase_survival[4]}%.
    For a total change of {total_survival_increased}%.
    
    Finally, the survival probability of the passenger is {round(base_survival + total_survival_increased, 2)}%.
    ''')
    
    result = {
        "result": f"\nYour passenger has a {surviving_prob}% probability of surviving!",
        "description": description
    }
    
    print("\nPrediction completed!")
    time.sleep(2)
    window.update_idletasks()

    return result

# Prediction of a fictional passenger
fictional_passenger = pd.DataFrame({
    "Pclass": [2],
    "Sex": [0],
    "Age": [26],
    "SibSp": [0],
    "Parch": [0]
})

# output = random_forest_predictor(fictional_passenger)
# print(output["result"])
# print(output["description"])
