from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# Import data
ruta_csv = Path(__file__).resolve().parents[0] / "DataSet_Titanic.csv"

df = pd.read_csv(ruta_csv)

# Save the column "Sobreviviente" as  "predictors"
predictors = df.drop("Sobreviviente", axis=1)

# Save the column "Sobreviviente" as  "data_to_predict"
data_to_predict = df["Sobreviviente"]

# Create base model
base_model = DecisionTreeClassifier(max_depth=2, random_state=42)

# Create model for GridSearchCV. As more "max_depth" the more accurate the model will be.
model = DecisionTreeClassifier(random_state=42)
parameters = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]}

# Implement GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='accuracy')

# Adjust GridSearchCV with our data
grid_search.fit(predictors, data_to_predict)

print(f"Better parameters: {grid_search.best_params_}")
print(f"Best score: {grid_search.best_score_:.4f}")

# We extract the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Train our models
base_model.fit(predictors, data_to_predict)
best_model.fit(predictors, data_to_predict)

# Make predictions over our data
prediction_base_model = base_model.predict(predictors)
prediction_best_model = best_model.predict(predictors)

# Compare accuracy of our models
print(f"Accuracy of the base model: {round(accuracy_score(data_to_predict, prediction_base_model)*100, 2)}%")
print(f"Accuracy of the GridSearch model: {round(accuracy_score(data_to_predict, prediction_best_model)*100, 2)}%")

choosen_model = ""
model_for_graphics = ""
if accuracy_score(data_to_predict, prediction_base_model) > accuracy_score(data_to_predict, prediction_best_model):
    print("The base model is better")
    choosen_model = prediction_base_model
    model_for_graphics = base_model
else:
    print("The GridSearch model is better")
    choosen_model = prediction_best_model
    model_for_graphics = best_model
    
# Create confusion matrix 
confusion_matrix(data_to_predict, choosen_model)

# Graphic the confusion matrix
ConfusionMatrixDisplay.from_estimator(model_for_graphics, predictors, data_to_predict, cmap=plt.cm.Blues, values_format='.2f')
plt.show()

# Graphic the confusion matrix (normalized)
ConfusionMatrixDisplay.from_estimator(model_for_graphics, predictors, data_to_predict, cmap=plt.cm.Blues, values_format='.2f', normalize="true")
plt.show()

# Graphic the tree
plt.figure(figsize=(10, 10))
tree.plot_tree(model_for_graphics, filled=True, feature_names=predictors.columns)
plt.show()

# Graphic the importances of the variables
# Create variables x (importance) e y (columns)
importance = model_for_graphics.feature_importances_
columns = predictors.columns

# Create graphic
sns.barplot(x=columns, y=importance)
plt.title("Importance of the variables")
plt.show()