from math import trunc
import streamlit as st
import pandas as pd
import pickle

# Create a Streamlit web app
st.title("Rain Fall Prediction")

# Add input fields for user input
st.sidebar.header("Input Features")

# Danh sách các tệp pickle chứa các mô hình
pickle_files = ['RandomForest.pkl', 'LinearRegression.pkl', 'CatBoost.pkl', 'XGBoost.pkl']

# Danh sách để lưu các mô hình đã tải
models = []
model_names = ['RandomForest', 'LinearRegression', 'CatBoost', 'XGBoost']

# Tải từng tệp pickle
for file in pickle_files:
    with open(file, 'rb') as model_file:
        model = pickle.load(model_file)
        models.append(model)

# Hàm để dự đoán và hiển thị kết quả từ tất cả các mô hình
def predict_and_display(df_pred):
    results = []
    for idx, model in enumerate(models):
        prediction_result = model.predict(df_pred)
        results.append((model_names[idx], prediction_result))
    
    for model_name, prediction in results:
         st.write(f"{model_name} prediction: {prediction}")

# Giao diện streamlit
year = st.sidebar.slider("Year", 2016, 2017, 2016)
month = st.sidebar.slider("Month", 1, 12, 1)
day = st.sidebar.slider("Day", 1, 31, 1)
hour = st.sidebar.slider("Hour", 0, 23, 0)
temperature = st.sidebar.slider("Temperature:", 0.0, 41.0, 0.0)
humidity = st.sidebar.slider("Humidity", 44, 0, 44)  # Sử dụng danh sách các giá trị từ 44 đến 100
wind_speed = st.sidebar.slider("Wind speed", 0, 18, 0)  # Sử dụng danh sách các giá trị từ 0 đến 18
wind_direction = st.sidebar.slider("Wind direction", 0, 360, 0)  # Sử dụng danh sách các giá trị từ 0 đến 3`60

# Tạo DataFrame từ các giá trị đầu vào
df_pred = pd.DataFrame({
    'Temperature': [temperature],
    'Humidity': [humidity],
    'Wind_Speed': [wind_speed],
    'Wind_Direction': [wind_direction],
    'Year': [year],
    'Month': [month],
    'Day': [day],
    'Hour': [hour]
})

if st.button('Predict'):
    predict_and_display(df_pred)