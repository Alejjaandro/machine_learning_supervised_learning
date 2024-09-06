from pathlib import Path
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import data
csv_path = Path(__file__).resolve().parents[0] / "DataSet_Titanic.csv"
df = pd.read_csv(csv_path)

# Split data: x = predictors, y = data_to_predict 
x = df.drop("Sobreviviente", axis=1)
y = df["Sobreviviente"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Create model
model = LogisticRegression(random_state=42)

# Train model
model.fit(x_train, y_train)

# Predict
y_pred = model.predict(x_test)

# Check accuracy
accuracy_score(y_test, y_pred)
print(f"Model accuracy: {round(accuracy_score(y_test, y_pred)*100, 2)}%")