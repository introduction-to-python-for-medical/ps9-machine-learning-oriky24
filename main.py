import pandas as pd
df = pd.read_csv('parkinsons.csv')
df = df.dropna()
df.head(10)
import seaborn as sns
sns.pairplot(df, hue='status')
X = df[['Shimmer:APQ3', 'MDVP:Fo(Hz)' ]]
y = df['status']
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
train_test_split(X_scaled, y, test_size=0.2, random_state=42
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=10)
knn_model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score

y_pred = knn_model.predict(X_val)
accuracy = accuracy_score(y_val, y_pred)

print(f"Accuracy: {accuracy}")

if accuracy < 0.8:
  print("Accuracy is below the threshold of 0.8. Try adjusting the model parameters or features.")
                 
