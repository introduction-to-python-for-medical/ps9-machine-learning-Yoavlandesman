# יבוא ספריות נדרשות
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# טעינת הנתונים והסרת ערכים חסרים
df = pd.read_csv('parkinsons.csv')
df = df.dropna()

# בחירת עמודות המאפיינים ועמודת המטרה
x = df[['MDVP:Fo(Hz)', 'MDVP:Jitter(%)']]
y = df['status']

# פיצול לנתוני אימון ונתוני בדיקה
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# נרמול הנתונים
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# בניית מודל KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train_scaled, y_train)

# חיזוי ובדיקת דיוק המודל
y_pred = knn.predict(x_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"דיוק המודל: {accuracy:.2f}")

