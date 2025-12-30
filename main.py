import pandas as pd
df = pd.read_csv('/content/parkinsons.csv')
df = df.dropna()
df.head()

X = df[['HNR', 'D2']]
y = df['status']

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.svm import SVC
svc = SVC()
svc.fit(x_train, y_train)

from sklearn.metrics import accuracy_score
y_pred = model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

import joblib
joblib.dump(svc, 'svc.joblib')
