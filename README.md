# Titanic Survival Prediction

This project is a Python-based application that predicts the survival of a fictional passenger aboard the Titanic based on user-provided characteristics.  

The application features a graphical user interface (GUI) built using [Tkinter](https://docs.python.org/es/3/library/tkinter.html#module-tkinter), and the prediction is powered by **3 machine learnings models** from [scikit-learn](https://scikit-learn.org/stable/) trained using the Titanic dataset.

## Features

- Graphical User Interface (GUI): The application provides an easy-to-use GUI for entering passenger details. Users can input passenger characteristics such as class, sex, age, number of siblings/spouses, and number of parents/children.

- Machine Learning Prediction: The prediction can carried out using 3 different ML models:
    * Gradient Boosting
    * Random Forest.
    * Tree Decision Classifier.

  All models estimates the probability of survival based on the user's input.

- Live Console Output: The interface also includes a console area that displays real-time messages, including the steps of the prediction process and any errors encountered. This allows users to observe the workflow and understand what is happening during the prediction.

- Real-time Feedback: When the user resizes the window, the font size of labels and entry fields adjusts dynamically to ensure an optimal user experience.

## Technical Details

All the models used for prediction come from the [scikit-learn](https://scikit-learn.org/stable/) library. It all have been trained on the same Titanic dataset to classify whether a passenger would have survived based on several input features.

- Data Scaling: In the **Gradient Boosting** model, features are scaled using StandardScaler from scikit-learn before training the model to improve performance and stability.

- [GridSearchCV](https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.GridSearchCV.html) for Hyperparameter Tuning: The model's hyperparameters have been optimized using GridSearchCV to select the best combination of parameters for better accuracy.

- SHAP Analysis: The application uses [SHAP (SHapley Additive exPlanations)](https://shap.readthedocs.io/en/latest/) values to understand the impact of each feature on the predicted probability of survival.

## How to Run the Project

### 1. Clone the Repository
```
git clone <repository-url>
cd titanic-survival-ui
```
### 2. Install the Dependencies

Ensure you have Python installed (version 3.7 or higher). Then, install the required Python packages using:
```
pip install -r requirements.txt
```
### 3. Run the Application

Execute the main script to launch the GUI:
```
python main.py
```
## User Guide

### Input Passenger Details

* Class: Enter the passenger's class (1 for First Class, 2 for Second Class, 3 for Third Class).

* Sex: Enter 0 for male or 1 for female.

* Age: Enter the age of the passenger.

* SibSp: Enter the number of siblings or spouses aboard the Titanic.

* Parch: Enter the number of parents or children aboard the Titanic.

### Prediction:

Click on the "Predict" button to see the predicted survival probability of the passenger.  
The result will be displayed as a message popup with all the info.

### Live Console

The console area at the bottom of the GUI displays messages about the progress of the prediction and any errors that may occur.

## Project Structure

`main.py` The main script that runs the GUI and handles user interactions.  

`console_function.py` Contains the function to send the console prints to the tkinter app.

`/Models` Directory where the 3 models and the Titanic Dataset are.
  - `DataSet_Titanic.csv`
  - `gradient_boosting_model.py`
  - `random_forest_model.py`
  - `tree_classifier_model.py`

`requirements.txt` Lists all dependencies required for the project.

## Dependencies

[Tkinter](https://docs.python.org/es/3/library/tkinter.html#module-tkinter): For creating the graphical user interface.

[Pandas](https://pandas.pydata.org/): For data manipulation and preprocessing.

[NumPy](https://numpy.org/): For numerical operations.

[Scikit-Learn](https://scikit-learn.org/stable/): For training the machine learning model and scaling data.

[SHAP](https://shap.readthedocs.io/en/latest/): For feature importance analysis.

## Future Improvements

- Add Additional Models: Integrate other machine learning models like Logistic Regression for comparison.

- Data Visualization: Provide visual feedback of feature importance using matplotlib or SHAP visual plots.

- Docker Support: Add Docker support for easier deployment and setup.

## License

This project is licensed under the Apache License. See the LICENSE file for more details.
