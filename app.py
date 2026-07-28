import streamlit as st
import pandas as pd
import joblib

# Load your pre-trained model directly from your repository folder structure
model = joblib.load('models/concrete_strength_model.pkl')

st.title("Concrete Strength Predictor")
st.write("Adjust the mix ingredients below to predict the concrete's compressive strength instantly.")

# Create input fields for the user based on dataset statistics
cement = st.slider("Cement (kg/m³)", 100, 550, 280)
slag = st.slider("Blast Furnace Slag (kg/m³)", 0, 360, 75)
fly_ash = st.slider("Fly Ash (kg/m³)", 0, 200, 50)
water = st.slider("Water (kg/m³)", 120, 250, 180)
superplasticizer = st.slider("Superplasticizer (kg/m³)", 0, 32, 6)
coarse_agg = st.slider("Coarse Aggregate (kg/m³)", 800, 1150, 970)
fine_agg = st.slider("Fine Aggregate (kg/m³)", 590, 1000, 770)
age = st.slider("Age (Days cured)", 1, 365, 28)

# Compile into a DataFrame matching your model's columns
input_data = pd.DataFrame(
    [[cement, slag, fly_ash, water, superplasticizer, coarse_agg, fine_agg, age]],
    columns=['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
             'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age (day)']
)

# Predict button
if st.button("Predict Strength"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Compressive Strength: {prediction:.2f} MPa")
