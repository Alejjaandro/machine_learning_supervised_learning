import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

def tree_predictor(passenger):
    # Import data
    df = pd.read_csv(Path(__file__).parent / "DataSet_Titanic.csv")

    # Save the column "Survived" as  "predictors"
    predictors = df.drop("Survived", axis=1)

    # Save the column "Survived" as  "data_to_predict"
    data_to_predict = df["Survived"]

    # Create base model
    base_model = DecisionTreeClassifier()

    # Create model for GridSearchCV. As more "max_depth" the more accurate the model will be.
    parameters = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]}

    # Implement GridSearchCV
    grid_search = GridSearchCV(estimator=base_model, param_grid=parameters, cv=5, scoring='accuracy')

    # Adjust GridSearchCV with our data
    grid_search.fit(predictors, data_to_predict)

    # We extract the best model from GridSearchCV
    grid_model = grid_search.best_estimator_

    # Train our models
    print("Training model....\n")
    base_model.fit(predictors, data_to_predict)
    grid_model.fit(predictors, data_to_predict)

    # Make predictions over our data
    prediction_base_model = base_model.predict(predictors)
    prediction_grid_model = grid_model.predict(predictors)

    # Compare accuracy of our models
    # print(f"Accuracy of the base model: {round(accuracy_score(data_to_predict, prediction_base_model)*100, 2)}%")
    # print(f"Accuracy of the GridSearch model: {round(accuracy_score(data_to_predict, prediction_grid_model)*100, 2)}%")

    # Select the best model
    choosen_model = ""
    if accuracy_score(data_to_predict, prediction_base_model) > accuracy_score(data_to_predict, prediction_grid_model):
        # print("The base model is better")
        choosen_model = base_model
    else:
        # print("The GridSearch model is better")
        choosen_model = grid_model
    
    importance_df = pd.DataFrame({
    'Variable': predictors.columns,
    'Importance': (base_model.feature_importances_ * 100).round(2)
    })

    # Fictional passenger prediction (Class, Genrer, Age, SiblingsSpouses, FatherSons)
    prediccion_ficticial = choosen_model.predict(passenger)
    if prediccion_ficticial == 1:
        return {
            "result":"Your passenger will survive!",
            "importance": importance_df
            }
    else:
        return {
            "result":"Your passenger will not survive!",
            "importance": importance_df
            }


# # Graphic the importances of the variables

# importance_df = importance_df.sort_values(by='Importance', ascending=False)

# plt.figure(figsize=(10, 6))
# plt.barh(importance_df['Variable'], importance_df['Importance'], color='royalblue')
# plt.xlabel('Importance')
# plt.title('Importance of the variables')
# plt.gca().invert_yaxis()  # Invert the y-axis to show the most important features first
# plt.show()

# ======================================== Graphics ========================================
# # Create confusion matrix 
# confusion_matrix(data_to_predict, choosen_model)

# # Graphic the confusion matrix
# ConfusionMatrixDisplay.from_estimator(model_for_graphics, predictors, data_to_predict, cmap=plt.cm.Blues, values_format='.2f')
# plt.show()

# # Graphic the confusion matrix (normalized)
# ConfusionMatrixDisplay.from_estimator(model_for_graphics, predictors, data_to_predict, cmap=plt.cm.Blues, values_format='.2f', normalize="true")
# plt.show()

# # Graphic the importances of the variables
# # Create variables x (importance) e y (columns)
# importance = model_for_graphics.feature_importances_
# columns = predictors.columns

# # Create graphic
# sns.barplot(x=columns, y=importance)
# plt.title("Importance of the variables")
# plt.show()