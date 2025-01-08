import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv('parkinsons.csv')
df = df.dropna()

features = ["Shimmer:APQ3", "MDVP:Fo(Hz)"]
X = df[features]
y = df['status']

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train, y_train)

y_pred = knn_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)
print(f"Accuracy: {accuracy}")

joblib.dump(knn_model, 'model.joblib')
joblib.dump(scaler, 'scaler.joblib')
