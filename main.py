import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


!wget https://raw.githubusercontent.com/yotam-biu/ps9/main/parkinsons.csv -O parkinsons.csv


df = pd.read_csv('parkinsons.csv')

x = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)


knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_scaled, y_train)

y_pred = knn.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"דיוק המודל: {accuracy:.2f}")

joblib.dump(knn, 'knn_model.joblib')
