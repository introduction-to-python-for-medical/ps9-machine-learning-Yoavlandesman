import pandas as pd

df = pd.read_csv('parkinsons.csv')
df = df.dropna()

x = df['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']
y = df['status']

import numpy as np
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
x_scale = scaler.fit_transform(x)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_scale, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_scale, y)
from sklearn.metrics import accuracy_score

y_pred = knn.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

