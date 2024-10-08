# Función para redirigir la consola a un widget de Tkinter
import tkinter as tk

class RedirectText:
    def __init__(self, text_widget):
        self.output = text_widget

    def write(self, string):
        self.output.insert(tk.END, string)
        self.output.see(tk.END)  # Hacer scroll automático al final

    def flush(self):  # Necesario para redirigir correctamente stdout
        pass