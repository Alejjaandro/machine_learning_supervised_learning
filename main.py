import os
import pandas as pd
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
        

titanic_predictor()