import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import os

# Load dataset
data = pd.read_csv("hi.csv")

# Handle missing values
data.dropna(inplace=True)

# Split features & target
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=50)

# Train Model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save Model
model_path = os.path.join(os.getcwd(), "model.pkl")
pickle.dump(model, open(model_path, "wb"))

# Test Accuracy
accuracy = model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")

# Example Prediction
new_features = [[8.9, 1.6, 670, 7, 5.5, 0, 5]]  # Ensure this matches feature count
predicted_dis = model.predict(new_features)
print("Predicted Disease:", predicted_dis[0])