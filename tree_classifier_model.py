from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn import tree

# Cargamos el dataset
ruta_csv = Path(__file__).resolve().parents[0] / "DataSet_Titanic.csv"

df = pd.read_csv(ruta_csv)

# Guardamos los atributos predictores (todas las etiquetas excepto "Sobreviviente")
datos_predictores = df.drop("Sobreviviente", axis=1)

# Y la etiqueta a predecir ("Sobreviviente")
dato_a_predecir = df["Sobreviviente"]

# Creamos un modelo de decision: cuanto mas profundo sea, mas precision obtendrá
modelo = DecisionTreeClassifier(random_state=42)
parametros = {"max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]}

# Implementamos GridSearchCV
grid_search = GridSearchCV(estimator=modelo, param_grid=parametros, cv=5, scoring='accuracy')

# Ajustamos GridSearchCV con los datos
grid_search.fit(datos_predictores, dato_a_predecir)

print(f"Mejores parámetros: {grid_search.best_params_}")
print(f"Mejor precisión: {grid_search.best_score_:.4f}")

# Obtenemos el mejor modelo que obtuvo GridSearchCV.
# Este es por defecto un DecisionTreeClassifier con los mejores parámetros encontrados.
mejor_modelo = grid_search.best_estimator_

# Creamos un modelo de decision con nuestros parámetros
arbol = DecisionTreeClassifier(max_depth=2, random_state=42)

# Entrenamos a la máquina
arbol.fit(datos_predictores, dato_a_predecir)
mejor_modelo.fit(datos_predictores, dato_a_predecir)

# Predecimos sobre nuestro set
prediccion_arbol = arbol.predict(datos_predictores)
prediccion_modelo = mejor_modelo.predict(datos_predictores)

# Comaparamos con las etiquetas reales
print(f"Exactitud del arbol: {round(accuracy_score(dato_a_predecir, prediccion_arbol)*100, 2)}%")
print(f"Exactitud del modelo: {round(accuracy_score(dato_a_predecir, prediccion_modelo)*100, 2)}%")

modelo_elegido = ""
modelo_para_graficar = ""
if accuracy_score(dato_a_predecir, prediccion_arbol) > accuracy_score(dato_a_predecir, prediccion_modelo):
    print("El arbol es mejor")
    modelo_elegido = prediccion_arbol
    modelo_para_graficar = arbol
else:
    print("El modelo es mejor")
    modelo_elegido = prediccion_modelo
    modelo_para_graficar = mejor_modelo
    
# Creamos una matriz de confusión con el modelo elejido
confusion_matrix(dato_a_predecir, modelo_elegido)

# Creamos un gráfico para la matriz de confusión
ConfusionMatrixDisplay.from_estimator(modelo_para_graficar, datos_predictores, dato_a_predecir, cmap=plt.cm.Blues, values_format='.2f')
plt.show()

# Creamos un gráfico para la matriz de confusión normalizada
ConfusionMatrixDisplay.from_estimator(modelo_para_graficar, datos_predictores, dato_a_predecir, cmap=plt.cm.Blues, values_format='.2f', normalize="true")
plt.show()

# Mostramos un árbol gráficamente
plt.figure(figsize=(10, 10))
tree.plot_tree(modelo_para_graficar, filled=True, feature_names=datos_predictores.columns)
plt.show()

# Hacemos un grafico con la importancia de las variables
# Creamos las variables x (importancias) e y (columnas)
importancias = modelo_para_graficar.feature_importances_
columnas = datos_predictores.columns

# Creamos el gráfico
sns.barplot(x=columnas, y=importancias)
plt.title("Importancia de las variables")
plt.show()