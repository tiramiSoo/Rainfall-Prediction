import streamlit as st
import pandas as pd
import numpy as np
import datetime
import pickle

# Create a Streamlit web app
st.title("Rain Fall Prediction")

# Add input fields for user input
st.sidebar.header("Input Features")

# Danh sách các tệp pickle chứa các mô hình
pickle_files = ['GRU.pkl', 'LinearRegression.pkl', 'LSTM.pkl', 'RNN.pkl']

# Danh sách để lưu các mô hình đã tải
models = []
model_names = ['GRU', 'Linear Regression', 'LSTM', 'RNN']

# Tải từng tệp pickle
for file in pickle_files:
    with open(file, 'rb') as model_file:
        model = pickle.load(model_file)
        models.append(model)

# Tạo các đặc trưng cho 24 bước thời gian
input_data = []
for t in range(24):
    # Đặt nhãn động cho ngày và giờ dựa trên bước thời gian
    date_input = st.sidebar.date_input(f"Date (t{t + 1}):", datetime.date(2016, 1, 15))
    time_input = st.sidebar.time_input(f"Time (t{t + 1}):", datetime.time(0, 0))

    # Trích xuất năm, tháng, ngày, và giờ từ date_input và time_input
    year = date_input.year
    month = date_input.month
    day = date_input.day
    hour = time_input.hour

    st.sidebar.subheader(f"Time Step {t + 1}")
    rain_fall = float(st.sidebar.text_input(f"Rainfall (t{t + 1}):", "0.0"))
    temperature = st.sidebar.slider(f"Temperature (t{t + 1}):", 0.0, 41.0, 0.0)
    humidity = st.sidebar.slider(f"Humidity (t{t + 1}):", 44, 100, 44)
    wind_speed = st.sidebar.slider(f"Wind speed (t{t + 1}):", 0, 18, 0)
    wind_direction = st.sidebar.slider(f"Wind direction (t{t + 1}):", 0, 360, 0)
    
    # Chuyển đổi hướng gió
    wd_rad = wind_direction * np.pi / 180
    Wind_x = wind_speed * np.cos(wd_rad)
    Wind_y = wind_speed * np.sin(wd_rad)

    input_data.append([year, month, day, hour, rain_fall, temperature, humidity, Wind_x, Wind_y])

# Convert to DataFrame
df_pred = pd.DataFrame(input_data, columns=['Rain fall', 
                                            'Temperature', 
                                            'Humidity', 
                                            'Wind_X', 
                                            'Wind_y', 
                                            'year', 
                                            'month', 
                                            'day', 
                                            'hour'])

# Add cyclic features
df_pred['datetime'] = pd.to_datetime(f"{year}-{month}-{day} {hour}:00:00")
start_time = pd.Timestamp(year=2016, month=1, day=15, hour=0)
df_pred['timestamp_h'] = (df_pred['datetime'] - start_time).dt.total_seconds() / 3600

day_hours = 24
year_hours = 365.2425 * day_hours

df_pred['Day_sin'] = np.sin(df_pred['timestamp_h'] * (2 * np.pi / day_hours))
df_pred['Day_cos'] = np.cos(df_pred['timestamp_h'] * (2 * np.pi / day_hours))
df_pred['Year_sin'] = np.sin(df_pred['timestamp_h'] * (2 * np.pi / year_hours))
df_pred['Year_cos'] = np.cos(df_pred['timestamp_h'] * (2 * np.pi / year_hours))

# Chọn các đặc tính cần thiết cho dự đoán
features = ['Rain fall', 'Temperature', 'Humidity', 'Wind_X', 'Wind_y', 'Day_sin', 'Day_cos', 'Year_sin', 'Year_cos']

# Bộ chọn mô hình
selected_model_name = st.sidebar.selectbox("Select Model", model_names)

# Hàm dự đoán và hiển thị kết quả
def predict_and_display_selected_model(df_pred, model_name):
    # Lấy mô hình đã chọn
    model_index = model_names.index(model_name)
    model = models[model_index]
    
    # Chuẩn bị dữ liệu đầu vào cho mô hình
    input_data = df_pred[features].values
    input_data = np.expand_dims(input_data, axis=0)  # Thêm chiều để có hình dạng (1, 24, 9)
    
    # Convert input_data to float32
    input_data = input_data.astype(np.float32)
    
    # Check for NaN values in input_data and handle them if necessary
    if np.isnan(input_data).any():
        nan_mean = np.nanmean(input_data, axis=0)
        inds = np.where(np.isnan(input_data))
        input_data[inds] = np.take(nan_mean, inds[1])
    
    # Dự đoán
    prediction = model.predict(input_data)
    
    # Hiển thị kết quả dự đoán
    # Chia 24 đầu ra thành 3 phần, mỗi phần có 8 đầu ra
    part_size = len(prediction[0]) // 3  # Tính kích thước mỗi phần

    # Tạo 3 cột
    col1, col2, col3 = st.columns(3)

    # Đầu ra cho cột 1
    with col1:
        for i in range(part_size):
            st.write(f"Prediction for {model_name} at time step {i+1}: {prediction[0][i][0]} (mm/h)")

    # Đầu ra cho cột 2
    with col2:
        for i in range(part_size, part_size*2):
            st.write(f"Prediction for {model_name} at time step {i+1}: {prediction[0][i][0]} (mm/h)")

    # Đầu ra cho cột 3
    with col3:
        for i in range(part_size*2, len(prediction[0])):
            st.write(f"Prediction for {model_name} at time step {i+1}: {prediction[0][i][0]} (mm/h)")

if st.button('Predict'):
    predict_and_display_selected_model(df_pred, selected_model_name)
