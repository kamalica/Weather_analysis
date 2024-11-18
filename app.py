#Machine learning
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import pickle

# Load the dataset
data = pd.read_csv("seattle-weather.csv")

# Drop the 'date' column
data = data.drop('date', axis=1)

# Encode the 'weather' column (categorical to numerical)
label_encoder = LabelEncoder()
data['weather_index'] = label_encoder.fit_transform(data['weather'])

# Define the features and target
X = data[['temp_min', 'wind', 'precipitation', 'weather_index']]
y = (data['temp_max'] > data['temp_max'].mean()).astype(int)  # Binary target above/below mean temp_max

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the SVM model
svm = SVC(kernel='linear', probability=True, random_state=42)
svm.fit(X_train, y_train)

# Evaluate the model
y_pred_proba = svm.predict_proba(X_test)[:, 1]
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC AUC for SVM: {roc_auc}")

# Save the trained model and scaler to pickle files
with open("svm_model.pkl", "wb") as model_file:
    pickle.dump(svm, model_file)

with open("scaler.pkl", "wb") as scaler_file:
    pickle.dump(scaler, scaler_file)

print("Model and scaler saved as pickle files.")

#deployment
import streamlit as st
import pickle
import numpy as np

# Load the trained model and scaler
with open("svm_model.pkl", "rb") as model_file:
    svm_model = pickle.load(model_file)

with open("scaler.pkl", "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

# Streamlit App
st.title("Weather Prediction App")
st.write("Predict whether the maximum temperature will be above or below the threshold using SVM.")

# User Inputs
temp_min = st.number_input("Minimum Temperature", value=0.0, step=0.1)  # Default value set to 0.0
wind = st.number_input("Wind Speed", value=0.0, step=0.1)  # Default value set to 0.0
precipitation = st.number_input("Precipitation", value=0.0, step=0.1)  # Default remains 0.0
weather_index = st.number_input(
    "Weather Index (choose: Sunny=0, Foggy=1, Rainy=2)", value=0, step=1, min_value=0, max_value=2
)


# Predict Button
if st.button("Predict"):
    # Prepare the input data
    input_data = np.array([[temp_min, wind, precipitation, weather_index]])
    scaled_data = scaler.transform(input_data)

    # Make prediction
    prediction_proba = svm_model.predict_proba(scaled_data)[0, 1]
    prediction = svm_model.predict(scaled_data)[0]

    # Display prediction
    if prediction == 1:
        st.success(f"The model predicts that the maximum temperature will be ABOVE the threshold (Probability: {prediction_proba:.2f}).")
    else:
        st.warning(f"The model predicts that the maximum temperature will be BELOW the threshold (Probability: {prediction_proba:.2f}).")
