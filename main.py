import os
import sys
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from tkinter.font import Font as tkFont
from tkinter.scrolledtext import ScrolledText

from Models.tree_classifier_model import tree_predictor
from Models.random_forest_model import random_forest_predictor
from Models.gradient_boosting_model import gradient_boosting_predictor
from console_function import RedirectText
from tutorial_window import show_tutorial

os.system('cls')

def predict_survival(window):    
    try:
        # Check if a model has been selected
        model = selected_model.get()
        if model == "None":
            messagebox.showwarning("Model Selection", "Please select a machine learning model.")
        else:
            print(f"Predicting using {model} model")
              
        # Obtain the values from the input fields
        pclass = int(class_var.get())
        sex = int(sex_var.get())
        age = int(age_var.get())
        sibsp = int(sibsp_var.get())
        parch = int(parch_var.get())
        
        # Create a DataFrame with the values
        passenger = pd.DataFrame({
            "Class": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "ParCh": [parch]
        })
        
        # Make the prediction
        if model == "Decision Tree":
            output = tree_predictor(passenger, window)
        elif model == "Random Forest":
            output = random_forest_predictor(passenger, window)
        elif model == "Gradient Boosting":
            output = gradient_boosting_predictor(passenger, window)
            
        messagebox.showinfo("\nPrediction", f"Result: {output['result']}\n\nDescription: {output['description']}")
         
    except Exception as e:
        messagebox.showerror("Error", f"There has been an error: {e}")

# Create the window and set its properties
window = tk.Tk()
window.minsize(500, 700)
window.maxsize(900, 900)
window.title("Titanic Survival Predictor")
window.config(padx=10, pady=10)

# Font
default_font = tkFont(family="Helvetica", size=14)

# Create a frame for the input fields
input_frame = tk.Frame(window)
input_frame.pack(pady=10, padx=10, fill='x')

# Labels and input fields
tk.Label(input_frame, text="Class (1, 2, 3):", font=default_font).grid(row=0, column=0, sticky='w', pady=5)
class_var = tk.Entry(input_frame, font=default_font)
class_var.grid(row=0, column=1)

tk.Label(input_frame, text="Sex (0=M, 1=F):", font=default_font).grid(row=1, column=0, sticky='w', pady=5)
sex_var = tk.Entry(input_frame, font=default_font)
sex_var.grid(row=1, column=1)

tk.Label(input_frame, text="Age:", font=default_font).grid(row=2, column=0, sticky='w', pady=5)
age_var = tk.Entry(input_frame, font=default_font)
age_var.grid(row=2, column=1)

tk.Label(input_frame, text="SibSp (Siblings/Spouses):", font=default_font).grid(row=3, column=0, sticky='w', pady=5)
sibsp_var = tk.Entry(input_frame, font=default_font)
sibsp_var.grid(row=3, column=1)

tk.Label(input_frame, text="ParCh (Parents/Childrens):", font=default_font).grid(row=4, column=0, sticky='w', pady=5)
parch_var = tk.Entry(input_frame, font=default_font)
parch_var.grid(row=4, column=1)

# Create a frame for the model selection
model_frame = tk.Frame(window)
model_frame.pack(pady=10, padx=10, fill='x')

# Variable to store the selected model
selected_model = tk.StringVar(value="None")

# Radio buttons to select the model
tk.Label(model_frame, text="Select a Machine Learning Model:", font=default_font).pack(anchor='w')
tk.Radiobutton(model_frame, text="Decision Tree", variable=selected_model, value="Decision Tree", font=default_font).pack(anchor='w')
tk.Radiobutton(model_frame, text="Random Forest", variable=selected_model, value="Random Forest", font=default_font).pack(anchor='w')
tk.Radiobutton(model_frame, text="Gradient Boosting", variable=selected_model, value="Gradient Boosting", font=default_font).pack(anchor='w')

# Button to predict
predict_button = tk.Button(window, text="Predict", command=lambda: predict_survival(window), font=default_font)
predict_button.pack(pady=10)

# Console area to show the output
console_frame = tk.Frame(window, bg="black")
console_frame.pack(pady=10, padx=10, fill='both', expand=True)

console_area = ScrolledText(console_frame, height=10, state='normal', wrap='word', font=12, bg="black", fg="white")
console_area.pack(fill='both', expand=True)

# Redirect the console output to the console area
sys.stdout = RedirectText(console_area)

# Start the window
show_tutorial()
window.mainloop()
