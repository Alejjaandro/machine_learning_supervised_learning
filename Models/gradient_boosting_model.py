from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

import pandas as pd
import shap
import numpy as np
from pathlib import Path
import textwrap
import math

def gradient_boosting_predictor(passenger):
    df = pd.read_csv(Path(__file__).parent / "DataSet_Titanic.csv")
    
    # Divide data into X and Y. X is the data we will use to predict Y
    X = df.drop("Survived", axis=1)
    Y = df["Survived"]

    # Divide data into validation and training data
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2, random_state=89)

    # Scaling the data
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_valid_scaled = pd.DataFrame(scaler.transform(X_valid), columns=X_valid.columns)
    
    # Initialize the model
    base_model = GradientBoostingClassifier(random_state=89)

    # Define the hyperparameters to tune
    parameters = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.01, 0.1, 0.2],
        "max_depth": [1, 2, 3, 4, 5, None],
    }
    
    # Create the GridSearchCV object
    grid_search = GridSearchCV(estimator=base_model, param_grid=parameters, cv=5, scoring='accuracy')
    
    print("\nTraining GridSearchCV...")
    grid_search.fit(X_train_scaled, Y_train)
    grid_model = grid_search.best_estimator_
    
    prediction_grid_model = grid_model.predict(X_valid_scaled)
    grid_accuracy = accuracy_score(Y_valid, prediction_grid_model)
    print(f"Accuracy of the GridSearch model on validation set: {round(grid_accuracy * 100, 2)}%\n")
    # print("Classification report:")
    # print(classification_report(Y_valid, prediction_grid_model))


    # Predicting the survival probability of the passenger
    passenger_scaled = pd.DataFrame(scaler.transform(passenger), columns=passenger.columns)
    surviving_prob = grid_model.predict_proba(passenger_scaled)[0][1]

    # SHAP values for each variable of the passenger
    explainer = shap.Explainer(grid_model, X_train_scaled)
    shap_values = explainer(passenger_scaled)

    shap_variable_values = shap_values.values[0]
    shap_base_value = shap_values.base_values[0]
    
    # Using log-odds to calculate the increase in survival probability
    base_probability = 1 / (1 + math.exp(-shap_base_value))
    impact_probabilities = []

    for shap_value in shap_variable_values:
        impact_log_odds = shap_base_value + shap_value
        impact_probability = 1 / (1 + math.exp(-impact_log_odds))
        impact_probabilities.append(round((impact_probability - base_probability) * 100, 2))
    
    total_impact_probability = round(sum(impact_probabilities), 2)
    
    shap_sum = np.sum(shap_variable_values)
    log_odds = shap_base_value + shap_sum
    final_probability = 1 / (1 + math.exp(-log_odds))

    description = textwrap.dedent(f'''
    Your have a base survival probability of {base_probability * 100:.2f}%.
    
    Each variable increases or decreases the survival probability of the passenger:
    - "Class" changes the survival probability by {impact_probabilities[0]}%.
    - "Sex" changes the survival probability by {impact_probabilities[1]}%.
    - "Age" changes the survival probability by {impact_probabilities[2]}%.
    - "SibSp" changes the survival probability by {impact_probabilities[3]}%.
    - "ParCh" changes the survival probability by {impact_probabilities[4]}%.
    For a total changes of {total_impact_probability}%.
    
    Finally, the survival probability of the passenger is {final_probability * 100:.2f}%.
    ''')
    
    result = {
        "result": f"\nYour passenger has a {surviving_prob:.2%} probability of surviving!",
        "description": description
    }
    
    return result

passenger = pd.DataFrame({
    "Class": [3],
    "Sex": [1],
    "Age": [26],
    "SibSp": [0],
    "ParCh": [0]
})

# output = gradient_boosting_predictor(passenger)
# print(output["result"])
# print(output["description"])
