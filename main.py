import os
import pandas as pd
import tkinter as tk
from tkinter import messagebox

from Models.tree_classifier_model import tree_predictor
from Models.random_forest_model import random_forest_predictor
from Models.gradient_boosting_model import gradient_boosting_predictor

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
        
def predict_survival():
    try:
        # Obtener los datos ingresados por el usuario
        pclass = int(class_var.get())
        sex = int(sex_var.get())
        age = int(age_var.get())
        sibsp = int(sibsp_var.get())
        parch = int(parch_var.get())
        
        # Crear un DataFrame para el pasajero
        passenger = pd.DataFrame({
            "Class": [pclass],
            "Sex": [sex],
            "Age": [age],
            "SibSp": [sibsp],
            "ParCh": [parch]
        })
        
        # Realizar la predicción
        output = random_forest_predictor(passenger)
        messagebox.showinfo("\nPredicción", f"Resultado: {output['result']}\n\nDescripción: {output['description']}")
         
    except Exception as e:
        messagebox.showerror("Error", f"Ha ocurrido un error: {e}")

# Crear la ventana principal de la interfaz
window = tk.Tk()
window.title("Predicción de Supervivencia del Titanic")

# Etiquetas y campos de entrada
tk.Label(window, text="Clase (1, 2, 3)").grid(row=0, column=0)
class_var = tk.StringVar()
tk.Entry(window, textvariable=class_var).grid(row=0, column=1)

tk.Label(window, text="Sexo (0=M, 1=F)").grid(row=1, column=0)
sex_var = tk.StringVar()
tk.Entry(window, textvariable=sex_var).grid(row=1, column=1)

tk.Label(window, text="Edad").grid(row=2, column=0)
age_var = tk.StringVar()
tk.Entry(window, textvariable=age_var).grid(row=2, column=1)

tk.Label(window, text="SibSp (Hermanos/Esposos)").grid(row=3, column=0)
sibsp_var = tk.StringVar()
tk.Entry(window, textvariable=sibsp_var).grid(row=3, column=1)

tk.Label(window, text="Parch (Padres/Hijos)").grid(row=4, column=0)
parch_var = tk.StringVar()
tk.Entry(window, textvariable=parch_var).grid(row=4, column=1)

# Botón para predecir
tk.Button(window, text="Predecir", command=predict_survival).grid(row=5, column=1)

# Iniciar la ventana
window.mainloop()