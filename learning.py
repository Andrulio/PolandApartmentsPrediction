import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

df = pd.read_csv("dataset/apartments_pl_2023_08.csv")
df = df.dropna()

df = df[df['city'].isin(
    ["bialystok", "bydgoszcz", "czestochowa", "gdansk", "gdynia", "katowice", "krakow", "lodz", "lublin", "poznan",
     "radom", "rzeszow", "szczecin", "warszawa", "wroclaw"])]
df = pd.get_dummies(df, columns=['city'], drop_first=True)

X = df[['squareMeters', 'rooms', 'floor', 'buildYear'] + [col for col in df.columns if col.startswith("city_")]]
y = df['price']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)

param_grid = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
}

grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(X_train, y_train)
model.fit(X_train, y_train)

print(f"Score: {model.score(X_test, y_test):.2f}")
with open("pkl/features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("pkl/model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("pkl/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
