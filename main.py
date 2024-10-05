import os
import pandas as pd
from Models.tree_classifier_model import tree_predictor
def titanic_predictor():
    end = False
    print("TITANIC SURVIVOR PREDICTOR")    
    
    while not end:
        print("Would you like to predict if you will survive the sinking of the Titanic?\n1. Yes\n2. No")
        
        if int(input()) == 1:
            os.system('cls')
            print("Welcome, please answer the following questions to predict if you will survive the sinking of the Titanic.")
            print("Choose your passenger class:\n1. First Class\n2. Second Class\n3. Third Class")
            passenger_class = int(input())
            print("Choose your gender:\n0. Female\n1. Male")
            sex = int(input())
            print("Choose your age")
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
            
            os.system('cls')
            
            print("="*50)
            print(f"YOUR PASSENGER: Class: {passenger_class}, Gender: {sex}, Age: {age}, SibSp: {sibsp}, ParCh: {parch}\n")
            print (tree_predictor(passenger))
            print("="*50)
            
            print("\nPress any key to continue...")
            input()
            os.system('cls')
        else:
            print("Ok, see you next time!")
            end = True
            return
        

titanic_predictor()