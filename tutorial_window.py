import tkinter as tk
from tkinter import Toplevel

def show_tutorial():
    # Create a new window
    tutorial_window = Toplevel()
    tutorial_window.title("Tutorial")
    tutorial_window.minsize(500, 700)
    tutorial_window.maxsize(900, 900)
    
    # Set background color and padding for a better visual
    tutorial_window.config(bg="white")

    # Text for the tutorial
    tutorial_message = """
    The RMS Titanic was one of the largest and most luxurious ships of its time, built to carry passengers from Europe to America.

    Here are some interesting facts about the Titanic:
    
        - Total number of passengers and crew on board: 2,224.
        - Number of survivors: Approximately 710 people.

    Ticket Prices:
    
        - First Class: $150 to $4,350 (equivalent to about $1,700 - $50,000 today).
        - Second Class: $60 (approximately $700 today).
        - Third Class: $15 to $40 (equivalent to about $170 - $460 today).

    The Titanic hit an iceberg on the night of April 14, 1912, and sank in the early hours of April 15. 
    Its legacy remains a lesson about human fragility and the importance of maritime safety.

    In this application, you can use historical passenger data to predict the survival probability of a fictional passenger. 

    This program uses machine learning models to predict the survival of a fictional passenger on the Titanic.

    You will need to provide the following information about the passenger:
    
    - Class (1, 2, 3): The class of the passenger.
    - Sex (0=M, 1=F): Your passenger sex.
    - Age: The age of the passenger.
    - SibSp (Siblings/Spouses): The number of siblings or spouses aboard.
    - ParCh (Parents/Children): The number of parents or children aboard.
    """

    # Creating a label for the title (bold)
    title_label = tk.Label(tutorial_window, text="Welcome to the Titanic Survival Predictor!", 
                           font=("Helvetica", 14, "bold"), bg="white", pady=10)
    title_label.pack()

    # Creating a label for the content
    tutorial_label = tk.Label(tutorial_window, text=tutorial_message, 
                              font=("Helvetica", 12), anchor="nw", justify="left", bg="white", padx=10, pady=10)
    tutorial_label.pack(expand=True, fill='both')

    # Add a close button
    close_button = tk.Button(tutorial_window, text="Close", command=tutorial_window.destroy, font=("Helvetica", 12))
    close_button.pack(pady=10)
