import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import tensorflow as tf
from datetime import date, timedelta
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

st.set_page_config(page_title="Dự Báo Cổ Phiếu AI", layout="wide")
st.title("📈 Ứng Dụng Dự Báo Giá Cổ Phiếu Bằng AI (LSTM)")

# Khung nhập mã cổ phiếu
ticker = st.text_input("Nhập mã cổ phiếu (VD: VNM.VN, FPT.VN, AAPL):", "VNM.VN")

# Tạo 2 cột để đặt 2 ô chọn ngày cho đẹp mắt
col1, col2 = st.columns(2)

with col1:
    # Mặc định ngày bắt đầu là 1/1/2018
    start_date = st.date_input("🗓️ Chọn ngày bắt đầu:", value=date(2018, 1, 1))

with col2:
    # Mặc định ngày kết thúc là ngày hiện tại
    end_date = st.date_input("🗓️ Chọn ngày kết thúc:", value=date.today())

# Kiểm tra xem người dùng có chọn ngày bắt đầu lớn hơn ngày kết thúc không
if start_date >= end_date:
    st.error("Lỗi: Ngày bắt đầu phải diễn ra trước ngày kết thúc!")
else:
    if st.button("Tải dữ liệu & Chạy dự báo", type="primary"):
        with st.spinner("Hệ thống đang tải dữ liệu và huấn luyện mô hình (sẽ mất khoảng 1-2 phút)..."):
            
            # Cố định seed để ổn định kết quả
            seed = 42
            np.random.seed(seed)
            tf.random.set_seed(seed)
            random.seed(seed)
            
            # Tải dữ liệu với ngày linh hoạt do người dùng chọn
            df = yf.download(ticker, start=start_date.strftime("%Y-%m-%d"), end=end_date.strftime("%Y-%m-%d"))
            
            if df.empty:
                st.error(f"Không tìm thấy dữ liệu cho mã {ticker} trong khoảng thời gian này. Vui lòng kiểm tra lại!")
            else:
                # Xử lý dữ liệu
                data = df[['Close']].values
                
                # Cần đảm bảo có đủ dữ liệu để train mô hình (ít nhất > 60 ngày)
                if len(data) <= 60:
                    st.error("Khoảng thời gian bạn chọn quá ngắn. Vui lòng chọn khoảng thời gian dài hơn (ít nhất vài tháng) để AI có thể học!")
                else:
                    train_size = int(0.8 * len(data))
                    train_data = data[:train_size]
                    test_data = data[train_size:]
                    
                    scaler = MinMaxScaler()
                    train_scaled = scaler.fit_transform(train_data)
                    test_scaled = scaler.transform(test_data)
                    
                    def create_sequences(data, window_size=60):
                        X, y = [], []
                        for i in range(window_size, len(data)):
                            X.append(data[i-window_size:i])
                            y.append(data[i])
                        return np.array(X), np.array(y)
                        
                    X_train, y_train = create_sequences(train_scaled)
                    test_combined = np.vstack((train_scaled[-60:], test_scaled))
                    X_test, y_test = create_sequences(test_combined)
                    
                    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
                    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
                    
                    val_size = int(0.1 * len(X_train))
                    X_val = X_train[-val_size:]
                    y_val = y_train[-val_size:]
                    X_train_final = X_train[:-val_size]
                    y_train_final = y_train[:-val_size]
                    
                    # Xây dựng và train mô hình LSTM
                    model = Sequential()
                    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
                    model.add(LSTM(50))
                    model.add(Dense(1))
                    model.compile(optimizer='adam', loss='mean_squared_error')
                    
                    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
                    
                    history = model.fit(
                        X_train_final, y_train_final,
                        validation_data=(X_val, y_val),
                        epochs=10,
                        batch_size=32,
                        shuffle=False,   
                        callbacks=[early_stop],
                        verbose=0 
                    )
                    
                    test_pred = model.predict(X_test, verbose=0)
                    test_pred = scaler.inverse_transform(test_pred)
                    y_test_real = scaler.inverse_transform(y_test.reshape(-1,1))
                    
                    rmse = np.sqrt(mean_squared_error(y_test_real, test_pred))
                    mae = mean_absolute_error(y_test_real, test_pred)
                    r2 = r2_score(y_test_real, test_pred)
                    
                    # Hiển thị các chỉ số đánh giá
                    st.subheader("📊 Kết quả đánh giá mô hình")
                    st.write(f"Đánh giá trên tập dữ liệu từ: **{start_date.strftime('%d/%m/%Y')}** đến **{end_date.strftime('%d/%m/%Y')}**")
                    
                    col_m1, col_m2, col_m3 = st.columns(3)
                    col_m1.metric("RMSE", f"{rmse:.2f}")
                    col_m2.metric("MAE", f"{mae:.2f}")
                    col_m3.metric("R² Score", f"{r2:.2f}")
                    
                    # Dự báo 10 ngày tới
                    last_60_days = np.vstack((train_scaled, test_scaled))[-60:]
                    last_60_days = last_60_days.reshape(1, 60, 1)
                    future_predictions = []
                    
                    for i in range(10):
                        next_pred = model.predict(last_60_days, verbose=0)
                        future_predictions.append(next_pred[0,0])
                        last_60_days = np.append(
                            last_60_days[:,1:,:],
                            next_pred.reshape(1,1,1),
                            axis=1
                        )
                    
                    future_predictions = np.array(future_predictions).reshape(-1,1)
                    future_predictions = scaler.inverse_transform(future_predictions)
                    
                    # Hiển thị bảng dự báo 10 ngày
                    st.subheader(f"🚀 Dự báo giá 10 ngày tiếp theo cho {ticker}")
                    future_df = pd.DataFrame(future_predictions, columns=["Giá dự báo"], index=[f"Ngày {i+1}" for i in range(10)])
                    st.table(future_df.style.format("{:,.2f}"))

                    # Vẽ biểu đồ giá
                    st.subheader("📉 Biểu đồ giá thật và dự báo")
                    fig, ax = plt.subplots(figsize=(12,6))
                    ax.plot(y_test_real, label="Giá thật")
                    ax.plot(test_pred, label="Dự báo test")
                    future_x = range(len(y_test_real), len(y_test_real)+10)
                    ax.plot(future_x, future_predictions, marker='o', label="10 ngày tới")
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Vẽ biểu đồ Loss
                    st.subheader("📉 Biểu đồ Loss Train vs Validation")
                    fig_loss, ax_loss = plt.subplots(figsize=(10,5))
                    ax_loss.plot(history.history['loss'], label='Train Loss')
                    ax_loss.plot(history.history['val_loss'], label='Validation Loss')
                    ax_loss.set_title("Loss Train vs Validation")
                    ax_loss.legend()
                    st.pyplot(fig_loss)
                    
                    st.success("Hoàn thành phân tích!")
