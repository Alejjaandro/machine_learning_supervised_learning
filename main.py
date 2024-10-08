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

os.system('cls')

def print_answer(passenger_class, sex, age, sibsp, parch, output):
    
    print("="*100)
    print(f"YOUR PASSENGER: Class: {passenger_class}, Sex: {sex}, Age: {age}, SibSp: {sibsp}, ParCh: {parch}")
    
    print(output["result"])
    print("="*100)
    
    print("Do you want to know more about the prediction?")
    print("1. Yes\n2. No")
    
    if int(input()) == 1:
        print("="*100)   
        print(output["description"])
        print("="*100)

def titanic_predictor():
    end = False
    
    print("*"*100)
    print("TITANIC SURVIVOR PREDICTOR")    
    while not end:
        print("Would you like to predict if you will survive the sinking of the Titanic?\n1. Yes\n2. No")
        
        if int(input()) == 1:
            print("\nAnswer the following questions to predict if you will survive the sinking of the Titanic.")
            print("Choose your passenger class:\n1. First Class\n2. Second Class\n3. Third Class")
            passenger_class = int(input())
            print("Choose your gender:\n0. Male\n1. Female")
            sex = int(input())
            print("Your age")
            age = int(input())
            print("Choose your siblings and spouses on board")
            sibsp = int(input())
            print("Choose your parents and children on board")
            parch = int(input())
            
            passenger =pd.DataFrame({
                "Class": [passenger_class],
                "Sex": [sex],
                "Age": [age],
                "SibSp": [sibsp],
                "ParCh": [parch]
            })
            
            output = random_forest_predictor(passenger)
            print_answer(passenger_class, sex, age, sibsp, parch, output)
                            
            print("*"*100)
            
            print("\nPress any key to continue...")
            input()
        else:
            print("Ok, see you next time!")
            end = True
            return
        
def predict_survival(window):
    try:
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
        output = random_forest_predictor(passenger, window)
        messagebox.showinfo("\nPrediction", f"Result: {output['result']}\n\nDescription: {output['description']}")
         
    except Exception as e:
        messagebox.showerror("Error", f"There has been an error: {e}")

# Create the window and set its properties
window = tk.Tk()
window.minsize(500, 400)
window.maxsize(700, 500)
window.title("Titanic Survival Predictor")
window.config(padx=10, pady=10)

# Font
default_font = tkFont(family="Helvetica", size=12)

# Función para actualizar el tamaño de la fuente al redimensionar la ventana
def resize_fonts(event):
    # Cambiar el tamaño de la fuente proporcionalmente al tamaño de la ventana
    new_size = max(12, int(event.width / 50))
    default_font.configure(size=new_size)

# Vincular el redimensionamiento de la ventana al evento para ajustar el tamaño de las fuentes
window.bind("<Configure>", resize_fonts)

# Labels and input fields
tk.Label(window, text="Class (1, 2, 3)", font=default_font).grid(row=0, column=0)
class_var = tk.StringVar()
tk.Entry(window, textvariable=class_var, font=default_font).grid(row=0, column=1)

tk.Label(window, text="Sex (0=M, 1=F)", font=default_font).grid(row=1, column=0)
sex_var = tk.StringVar()
tk.Entry(window, textvariable=sex_var, font=default_font).grid(row=1, column=1)

tk.Label(window, text="Age", font=default_font).grid(row=2, column=0)
age_var = tk.StringVar()
tk.Entry(window, textvariable=age_var, font=default_font).grid(row=2, column=1)

tk.Label(window, text="SibSp (Siblings/Spouses)", font=default_font).grid(row=3, column=0)
sibsp_var = tk.StringVar()
tk.Entry(window, textvariable=sibsp_var, font=default_font).grid(row=3, column=1)

tk.Label(window, text="ParCh (Parents/Childrens)", font=default_font).grid(row=4, column=0)
parch_var = tk.StringVar()
tk.Entry(window, textvariable=parch_var, font=default_font).grid(row=4, column=1)

# Button to predict
tk.Button(window, text="Predict", command=lambda: predict_survival(window), font=default_font).grid(row=5, column=1, sticky="nsew")

# Área de mensajes (con Scroll)
console_area = ScrolledText(window, height=10, state='normal', wrap='word', font=default_font, bg="black", fg="white")
console_area.grid(row=6, column=0, columnspan=2, sticky="nsew", padx=10, pady=10)

# Redirigir stdout a la consola de Tkinter
sys.stdout = RedirectText(console_area)

# Settings so the window is resized correctly
for i in range(6):
    window.grid_rowconfigure(i, weight=1)

window.grid_columnconfigure(0, weight=1)
window.grid_columnconfigure(1, weight=1)

# Start the window
window.mainloop()