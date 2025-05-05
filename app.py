import streamlit as st
import pickle
import pandas as pd

with open("pkl/model.pkl", "rb") as f:
    model = pickle.load(f)
with open("pkl/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("pkl/features.pkl", "rb") as f:
    feature_names = pickle.load(f)

st.title("Przewidywane ceny mieszkań w Polsce 2023")

area = st.number_input("Powierzchnia (m²)", min_value=10, max_value=200, value=50)
rooms = st.selectbox("Liczba pokoi", [1, 2, 3, 4, 5])
floor = st.number_input("Piętro", min_value=0, max_value=30, value=2)
buildYear = st.number_input("Rok budowy", min_value=1900, max_value=2025, value=1980)

cities = ["bydgoszcz", "czestochowa", "gdansk", "gdynia", "katowice", "krakow", "lodz", "lublin", "poznan",
          "radom", "rzeszow", "szczecin", "warszawa", "wroclaw"]
district = st.selectbox("Miasto", ["bialystok"] + cities)

input_dict = {col: 0 for col in feature_names}
input_dict.update({
    'squareMeters': area,
    'rooms': rooms,
    'floor': floor,
    'buildYear': buildYear
})
if district in cities:
    input_dict[f"city_{district}"] = 1

input_df = pd.DataFrame([input_dict])
input_scaled = scaler.transform(input_df)

if st.button("Przewiduj cenę"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"Szacowana cena: {prediction:,.0f} zł")
