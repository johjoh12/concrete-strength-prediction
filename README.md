# Predicting Concrete Compressive Strength Using Ensemble Machine Learning
**Live Interactive Web App:** [Click here to use the app](https://concrete-strength-prediction-fi54cjzbsfq2djjddqqgi6.streamlit.app/)
**Watch the 3-Minute Project Presentation:** [Click here to watch on YouTube](YOUR_YOUTUBE_LINK_HERE)


## 📝 Abstract
In civil engineering, determining the compressive strength of concrete traditionally requires preparing mix samples and waiting for a destructive 28-day laboratory curing test. This project develops an end-to-end Machine Learning pipeline utilizing an ensemble method (Random Forest Regressor) to predict concrete compressive strength instantly based on its raw mix design ingredients and curing age. Using a historical dataset of 1,030 concrete formulations, the predictive model acts as an automated tool to optimize mix proportions and eliminate labor-intensive waiting periods.

---

## Repository Structure
* `data/` : Contains the experimental dataset (`Concrete Compressive Strength.csv`) featuring 1,030 mix formulations.
* `models/` : Stores the deployed, pre-trained model file (`concrete_strength_model.pkl`) generated via `joblib`.
* `concrete_strength_analysis.ipynb` : The complete executable Google Colab notebook showcasing data exploration, model training, evaluation, and custom interface testing.

---

## Dataset & Feature Analysis
The model analyzes 8 engineering features to predict 1 target output variable:

### Input Features (Predictors):
1. **Cement** (kg in a m³ mixture)
2. **Blast Furnace Slag** (kg in a m³ mixture)
3. **Fly Ash** (kg in a m³ mixture)
4. **Water** (kg in a m³ mixture)
5. **Superplasticizer** (kg in a m³ mixture)
6. **Coarse Aggregate** (kg in a m³ mixture)
7. **Fine Aggregate** (kg in a m³ mixture)
8. **Age** (Curing time duration in days, ranging from 1 to 365)

### Target Output:
* **Concrete Compressive Strength** (Measured in Megapascals, MPa)

---

## Model Performance & Evaluation Metrics
The ensemble tree model was evaluated on a distinct testing subset. The evaluation metrics demonstrate exceptional prediction accuracy, making it highly applicable for structural mix estimations:

* **R² Score (Coefficient of Determination):** `0.8843` (The model successfully accounts for **88.43%** of the variance in concrete strength).
* **Mean Absolute Error (MAE):** `3.75 MPa` (On average, the model's structural estimates deviate from actual laboratory destruction tests by less than 4 MPa).
* **Root Mean Squared Error (RMSE):** `5.46 MPa`

---

## 🔍 Key Engineering Insights (Feature Importance)
The non-linear tree model calculated the individual impact weights of each variable on final structural strength. The results align perfectly with established principles of concrete chemistry:

| Rank | Feature | Relative Importance Weight | Architectural Signification |
| :--- | :--- | :--- | :--- |
| 1 | **Age (day)** | 33.37% | Reflects the hydration process over time. |
| 2 | **Cement** | 32.32% | The primary binder; stronger correlation to solid paste. |
| 3 | **Water** | 12.55% | Critical negative linear contributor; dilution weakens bonds. |
| 4 | **Blast Furnace Slag** | 7.67% | Supplementary cementitious material alternative. |
| 5 | **Superplasticizer** | 5.84% | Chemical additive reducing water requirements. |
| 6 | **Fine Aggregate** | 3.51% | Sand filling structure voids. |
| 7 | **Coarse Aggregate** | 2.83% | Gravel elements providing bulk skeleton volume. |
| 8 | **Fly Ash** | 1.92% | Secondary environmental micro-filler ingredient. |

---

## How to Run Predictions Locally
To reload the pre-trained production model and calculate the expected strength of a brand new structural concrete recipe instantly, execute this snippet:

```python
import joblib
import pandas as pd

# 1. Bring the saved model back into memory
model = joblib.load('models/concrete_strength_model.pkl')

# 2. Enter custom mix recipes (Matches the exact order of features used during training)
sample_mixes_df = pd.DataFrame(
    [[280.0, 75.0, 50.0, 180.0, 6.0, 970.0, 770.0, 28],
     [400.0, 100.0, 0.0, 150.0, 12.0, 950.0, 750.0, 90]],
  columns=['Cement', 'Blast Furnace Slag', 'Fly Ash', 'Water', 
             'Superplasticizer', 'Coarse Aggregate', 'Fine Aggregate', 'Age (day)']
)

# 3. Compute the predicted strength values
predicted_strengths = model.predict(sample_mixes_df)

# 4. Print outputs for each recipe index
print("--- Sampled Concrete Strength Predictions ---")
print(f"Sample 1 (Standard 28-Day Mix) Predicted Strength: {predicted_strengths[0]:.2f} MPa")
print(f"Sample 2 (High-Strength 90-Day Mix) Predicted Strength: {predicted_strengths[1]:.2f} MPa")
```
## My Next Steps:
I started this project to practice working the scikit-Learn library and processing real-world engineering datasets. Next, I plan to deploy this model as a simple web app using Streamlit so users can plug in numbers via a browser interface instead of running code blocks.
