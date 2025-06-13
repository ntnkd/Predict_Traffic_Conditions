import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import ReduceLROnPlateau
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
from flask import Flask, request, jsonify, render_template_string
from flask_socketio import SocketIO, emit
import threading
import time
import os
import matplotlib.pyplot as plt
from collector import TomTomTrafficCollector

class TrafficPredictor:
    def __init__(self, initial_data_path, model_path=None):
        # Khởi tạo với đường dẫn dữ liệu ban đầu và mô hình đã lưu (nếu có)
        self.scaler = MinMaxScaler()
        self.sequence_length = 12
        self.features = ['speed', 'vehicle_density', 'time_of_day', 'day_of_week']
        self.speed_thresholds = {'very_slow': 20, 'slow': 40, 'moderate': 60, 'fast': float('inf')}
        self.density_thresholds = {'low': 30, 'medium': 60, 'high': float('inf')}
        self.data_path = initial_data_path
        self.original_data = pd.read_csv(initial_data_path)
        self.model = tf.keras.models.load_model(model_path) if model_path and os.path.exists(model_path) else None
        self.load_and_preprocess_data()
        self.realtime_data = pd.DataFrame()

    def load_and_preprocess_data(self):
        # Tiền xử lý dữ liệu ban đầu: đọc CSV, chuẩn hóa, chia tập huấn luyện và kiểm tra
        df = self.original_data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(['location_name', 'timestamp'])
        self.locations = df['location_name'].unique()
        data_scaled = self.scaler.fit_transform(df[self.features])
        
        self.location_data = {}
        for location in self.locations:
            location_mask = df['location_name'] == location
            location_data = data_scaled[location_mask]
            X, y = [], []
            for i in range(len(location_data) - self.sequence_length - 2):
                X.append(location_data[i:(i + self.sequence_length)])
                y.append(location_data[i + self.sequence_length + 2, :2])
            X = np.array(X)
            y = np.array(y)
            train_size = int(len(X) * 0.8)
            self.location_data[location] = {
                'X_train': X[:train_size], 'X_test': X[train_size:],
                'y_train': y[:train_size], 'y_test': y[train_size:],
                'raw_data': df[location_mask]
            }

    def build_model(self):
        # Xây dựng mô hình LSTM để dự đoán tốc độ và mật độ
        self.model = Sequential([
            LSTM(32, return_sequences=True, input_shape=(self.sequence_length, len(self.features))),
            Dropout(0.3),
            LSTM(16),
            Dropout(0.3),
            Dense(8, activation='relu', kernel_regularizer=l2(0.01)),
            Dense(2)
        ])
        optimizer = Adam(learning_rate=0.0005)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def train_model(self, epochs=200, batch_size=32):
        # Huấn luyện mô hình LSTM với dữ liệu ban đầu
        if not self.model:
            self.build_model()
        early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=0.00001)
        X_train = np.concatenate([data['X_train'] for data in self.location_data.values()])
        y_train = np.concatenate([data['y_train'] for data in self.location_data.values()])
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.legend()
        plt.savefig("loss_plot.png")
        return history

    def update_realtime_data(self, new_data: pd.DataFrame):
        # Cập nhật dữ liệu thời gian thực và lưu vào file CSV, không giới hạn số bản ghi
        if not new_data.empty:
            self.realtime_data = pd.concat([self.realtime_data, new_data], ignore_index=True)
            
            # Ghi dữ liệu mới vào file temp_traffic_data.csv
            if os.path.exists(self.data_path):
                existing_data = pd.read_csv(self.data_path)
                updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                if len(updated_data) > 10000:
                    updated_data = updated_data.iloc[-10000:]
                updated_data.to_csv(self.data_path, index=False)
            else:
                new_data.to_csv(self.data_path, index=False)

            # Cập nhật dữ liệu trong bộ nhớ cho từng vị trí
            for location in self.locations:
                location_mask = self.realtime_data['location_name'] == location
                self.location_data[location]['raw_data'] = pd.concat(
                    [self.location_data[location]['raw_data'], self.realtime_data[location_mask]]
                )

    def predict_realtime(self, location, minutes_ahead=10):
        # Dự đoán tốc độ và mật độ trong tương lai cho một vị trí
        location_data = self.location_data[location]
        latest_sequence = location_data['raw_data'].iloc[-self.sequence_length:][self.features]
        
        latest_sequence_df = pd.DataFrame(latest_sequence, columns=self.features)
        scaled_sequence = self.scaler.transform(latest_sequence_df)
        
        steps_per_5min = 1
        prediction_steps = int(minutes_ahead / 5) * steps_per_5min
        
        current_input = scaled_sequence[np.newaxis, :, :]
        predictions = []
        
        last_time = pd.to_datetime(location_data['raw_data'].iloc[-1]['timestamp'])
        
        for step in range(prediction_steps):
            pred = self.model.predict(current_input, verbose=0)
            predictions.append(pred[0])
            
            new_input = np.zeros((1, self.sequence_length, len(self.features)))
            new_input[0, :-1] = current_input[0, 1:]
            pred_adjusted = np.clip(pred[0], 0, 1)
            new_input[0, -1, :2] = pred_adjusted
            
            new_time = last_time + pd.Timedelta(minutes=5 * (step + 1))
            new_time_of_day = new_time.hour + new_time.minute / 60
            new_day_of_week = new_time.weekday()
            
            time_features_df = pd.DataFrame([[0, 0, new_time_of_day, new_day_of_week]], columns=self.features)
            time_features = self.scaler.transform(time_features_df)[0, 2:]
            new_input[0, -1, 2:] = time_features
            
            current_input = new_input
        
        predictions = np.array(predictions)
        full_predictions_df = pd.DataFrame(np.concatenate([predictions, np.zeros((len(predictions), 2))], axis=1), columns=self.features)
        predictions_inv = self.scaler.inverse_transform(full_predictions_df)[:, :2]
        predictions_inv[:, 1] = np.clip(predictions_inv[:, 1], 0, 100)
        
        return predictions_inv[-1]

    def evaluate_model(self):
        # Lấy dữ liệu kiểm tra từ tất cả các vị trí
        X_test = np.concatenate([data['X_test'] for data in self.location_data.values()])
        y_test = np.concatenate([data['y_test'] for data in self.location_data.values()])
        
        # Dự đoán bằng LSTM
        y_pred_scaled_lstm = self.model.predict(X_test, verbose=0)
        y_test_full = np.concatenate([y_test, np.zeros((len(y_test), 2))], axis=1)
        y_pred_full_lstm = np.concatenate([y_pred_scaled_lstm, np.zeros((len(y_pred_scaled_lstm), 2))], axis=1)
        
        y_test_real = self.scaler.inverse_transform(y_test_full)[:, :2]
        y_pred_real_lstm = self.scaler.inverse_transform(y_pred_full_lstm)[:, :2]
        
        # Chuẩn bị dữ liệu 3D cho LR và RF bằng cách làm phẳng mỗi chuỗi
        num_samples = X_test.shape[0]
        sequence_length = X_test.shape[1]  # 6
        num_features = X_test.shape[2]     # 4
        X_test_2d = X_test.reshape(num_samples, sequence_length * num_features)  # (samples, 24)
        
        y_test_speed = y_test_real[:, 0]    # Tốc độ thực tế
        y_test_density = y_test_real[:, 1]  # Mật độ thực tế
        
        # Huấn luyện và dự đoán bằng Linear Regression
        lr_speed = LinearRegression()
        lr_density = LinearRegression()
        lr_speed.fit(X_test_2d, y_test_speed)
        lr_density.fit(X_test_2d, y_test_density)
        y_pred_speed_lr = lr_speed.predict(X_test_2d)
        y_pred_density_lr = lr_density.predict(X_test_2d)
        
        
        # Huấn luyện và dự đoán bằng GRU
        model_gru = Sequential([
            GRU(50, activation='relu', input_shape=(X_test.shape[1], X_test.shape[2]), return_sequences=False),
            Dense(2)
        ])
        model_gru.compile(optimizer='adam', loss='mse')
        model_gru.fit(X_test, y_test, epochs=50, batch_size=32, verbose=0)
        y_pred_gru = model_gru.predict(X_test, verbose=0)
        y_pred_gru_real = self.scaler.inverse_transform(
            np.concatenate([y_pred_gru, np.zeros((len(y_pred_gru), 2))], axis=1)
        )[:, :2]
        
        # Tính toán các chỉ số đánh giá cho LSTM
        mae_speed_lstm = mean_absolute_error(y_test_speed, y_pred_real_lstm[:, 0])
        mse_speed_lstm = mean_squared_error(y_test_speed, y_pred_real_lstm[:, 0])
        rmse_speed_lstm = np.sqrt(mse_speed_lstm)
        
        mae_density_lstm = mean_absolute_error(y_test_density, y_pred_real_lstm[:, 1])
        mse_density_lstm = mean_squared_error(y_test_density, y_pred_real_lstm[:, 1])
        rmse_density_lstm = np.sqrt(mse_density_lstm)
        
        # Tính toán các chỉ số đánh giá cho LR
        mae_speed_lr = mean_absolute_error(y_test_speed, y_pred_speed_lr)
        mse_speed_lr = mean_squared_error(y_test_speed, y_pred_speed_lr)
        rmse_speed_lr = np.sqrt(mse_speed_lr)
        
        mae_density_lr = mean_absolute_error(y_test_density, y_pred_density_lr)
        mse_density_lr = mean_squared_error(y_test_density, y_pred_density_lr)
        rmse_density_lr = np.sqrt(mse_density_lr)
        

        
        # Tính toán các chỉ số đánh giá cho GRU
        mae_speed_gru = mean_absolute_error(y_test_speed, y_pred_gru_real[:, 0])
        mse_speed_gru = mean_squared_error(y_test_speed, y_pred_gru_real[:, 0])
        rmse_speed_gru = np.sqrt(mse_speed_gru)
        
        mae_density_gru = mean_absolute_error(y_test_density, y_pred_gru_real[:, 1])
        mse_density_gru = mean_squared_error(y_test_density, y_pred_gru_real[:, 1])
        rmse_density_gru = np.sqrt(mse_density_gru)
        
        # Tạo bảng so sánh
        comparison_data = {
            'Metric': ['MAE', 'MSE', 'RMSE'],
            'LSTM_Speed': [mae_speed_lstm, mse_speed_lstm, rmse_speed_lstm],
            'LR_Speed': [mae_speed_lr, mse_speed_lr, rmse_speed_lr],
            'GRU_Speed': [mae_speed_gru, mse_speed_gru, rmse_speed_gru],
            'LSTM_Density': [mae_density_lstm, mse_density_lstm, rmse_density_lstm],
            'LR_Density': [mae_density_lr, mse_density_lr, rmse_density_lr],
            'GRU_Density': [mae_density_gru, mse_density_gru, rmse_density_gru]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(2)
        
        print("\nModel Comparison Table:")
        print(comparison_df.to_string(index=False))
        
        return comparison_df

    def _get_speed_color(self, speed):
        # Trả về màu sắc dựa trên tốc độ
        if speed <= self.speed_thresholds['very_slow']:
            return 'green'
        elif speed <= self.speed_thresholds['slow']:
            return 'yellow'
        elif speed <= self.speed_thresholds['moderate']:
            return 'orange'
        else:
            return 'red'

    def _get_density_color(self, density):
        # Trả về màu sắc dựa trên mật độ
        if density <= self.density_thresholds['low']:
            return 'green'
        elif density <= self.density_thresholds['medium']:
            return 'yellow'
        else:
            return 'red'

    def _get_speed_category(self, speed):
        # Phân loại tốc độ thành các danh mục
        if speed <= self.speed_thresholds['very_slow']:
            return 'Very Slow'
        elif speed <= self.speed_thresholds['slow']:
            return 'Slow'
        elif speed <= self.speed_thresholds['moderate']:
            return 'Moderate'
        else:
            return 'Fast'

    def _get_density_category(self, density):
        # Phân loại mật độ thành các danh mục
        if density <= self.density_thresholds['low']:
            return 'Low'
        elif density <= self.density_thresholds['medium']:
            return 'Medium'
        else:
            return 'High'

    def create_traffic_map_data(self, minutes_ahead=10):
        # Tạo dữ liệu bản đồ giao thông cho Leaflet
        unique_locations = self.realtime_data.groupby('location_name').agg({
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index() if not self.realtime_data.empty else self.original_data.groupby('location_name').agg({
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        map_data = []
        for _, row in unique_locations.iterrows():
            location = row['location_name']
            pred = self.predict_realtime(location, minutes_ahead)
            current_data = self.location_data[location]['raw_data'].iloc[-1][['speed', 'vehicle_density']].values
            
            speed_color = self._get_speed_color(pred[0])
            density_color = self._get_density_color(pred[1])
            speed_cat = self._get_speed_category(pred[0])
            density_cat = self._get_density_category(pred[1])
            
            popup_content = f"""
            <div style="font-family: Arial, sans-serif; padding: 10px; min-width: 200px;">
                <h3 style="color: #2c3e50; margin-bottom: 10px;">{location}</h3>
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px; margin-bottom: 10px;">
                    <h4 style="margin: 0 0 5px 0; color: #2c3e50;">Current Status:</h4>
                    <div style="margin-bottom: 5px;">
                        <strong>Speed:</strong> 
                        <span style="color: {self._get_speed_color(current_data[0])}; -webkit-text-stroke: 1px rgba(0, 0, 0, 0.2);">
                            {current_data[0]:.1f} km/h ({self._get_speed_category(current_data[0])})
                        </span>
                    </div>
                    <div>
                        <strong>Density:</strong> 
                        <span style="color: {self._get_density_color(current_data[1])}; -webkit-text-stroke: 1px rgba(0, 0, 0, 0.2);">
                            {current_data[1]:.1f}% ({self._get_density_category(current_data[1])})
                        </span>
                    </div>
                </div>
                <div style="background-color: #f8f9fa; padding: 10px; border-radius: 5px;">
                    <h4 style="margin: 0 0 5px 0; color: #2c3e50;">Prediction (Next {minutes_ahead} minutes):</h4>
                    <div style="margin-bottom: 5px;">
                        <strong>Speed:</strong> 
                        <span style="color: {speed_color}; -webkit-text-stroke: 1px rgba(0, 0, 0, 0.2);">
                            {pred[0]:.1f} km/h ({speed_cat})
                        </span>
                    </div>
                    <div>
                        <strong>Density:</strong> 
                        <span style="color: {density_color}; -webkit-text-stroke: 1px rgba(0, 0, 0, 0.2);">
                            {pred[1]:.1f}% ({density_cat})
                        </span>
                    </div>
                </div>
            </div>
            """
            
            map_data.append({
                'location': location,
                'latitude': row['latitude'],
                'longitude': row['longitude'],
                'speed_color': speed_color,
                'density_color': density_color,
                'popup': popup_content,
                'tooltip': f"{location} - Click for details"
            })
        return map_data

app = Flask(__name__)
app.static_folder = '.'
socketio = SocketIO(app)

def initialize_predictor():
    # Khởi tạo predictor, huấn luyện mô hình mới và hiển thị bảng so sánh
    initial_data_path = 'temp_traffic_data.csv'
    
    print("Bắt đầu huấn luyện mô hình mới...")
    predictor = TrafficPredictor(initial_data_path)
    predictor.train_model(epochs=200, batch_size=32)
    print("Mô hình đã được huấn luyện.")
    
    predictor.evaluate_model()
    return predictor

predictor = initialize_predictor()
collector = TomTomTrafficCollector(api_key="8pAHAqpTkhGyvgvl1aWLq4nkbYGuM5Ld")
locations = [
    {'name': 'Dragon Bridge', 'latitude': 16.061025, 'longitude': 108.226179},
    {'name': 'Han River Bridge', 'latitude': 16.072118, 'longitude': 108.22656},
    {'name': 'Hue T-junction', 'latitude': 16.060478, 'longitude': 108.177008},
    {'name': 'Da Nang station', 'latitude': 16.070959, 'longitude': 108.209465},
]

# Khởi tạo lock
data_collection_lock = threading.Lock()

def background_task():
    thread_id = threading.get_ident()
    print(f"Background task running in thread {thread_id}")
    retrain_interval = 3600  # Retrain mỗi 1 giờ (3600 giây)
    last_retrain_time = time.time()
    min_new_records = 100  # Số bản ghi mới tối thiểu để retrain

    while True:
        with data_collection_lock:
            realtime_data = collector.get_realtime_data(locations)
            if not realtime_data.empty:
                predictor.update_realtime_data(realtime_data)
                map_data = predictor.create_traffic_map_data(minutes_ahead=10)
                socketio.emit('traffic_update', map_data)

                # Kiểm tra xem có nên huấn luyện lại mô hình không
                current_time = time.time()
                if (current_time - last_retrain_time >= retrain_interval) and len(predictor.realtime_data) >= min_new_records:
                    print("Đủ dữ liệu mới, bắt đầu huấn luyện lại mô hình...")
                    predictor.original_data = pd.read_csv(predictor.data_path)  # Cập nhật original_data từ file CSV
                    predictor.load_and_preprocess_data()  # Cập nhật lại dữ liệu đã tiền xử lý
                    predictor.train_model(epochs=50, batch_size=32)  # Huấn luyện với số epoch nhỏ hơn
                    predictor.evaluate_model()
                    print("Mô hình đã được huấn luyện lại.")
                    last_retrain_time = current_time

        time.sleep(300)  # Thu thập dữ liệu mỗi 5 phút

# Biến toàn cục để theo dõi trạng thái luồng
background_thread = None
thread_started = False  # Thêm biến cờ để kiểm tra
def start_background_task():
    global background_thread, thread_started
    if not thread_started:
        background_thread = threading.Thread(target=background_task)
        background_thread.daemon = True
        background_thread.start()
        thread_started = True
        print("Background task started.")
    else:
        print("Background task is already running.")

@app.route('/')
def index():
    # Trả về giao diện web
    with open('traffic_predictions_map.html', encoding='utf-8') as f:
        return render_template_string(f.read())
    
@app.route('/predict', methods=['POST'])
def predict():
    # Xử lý yêu cầu dự đoán từ client
    minutes_ahead = int(request.json.get('minutes', 10))
    map_data = predictor.create_traffic_map_data(minutes_ahead)
    return jsonify(map_data)

if __name__ == "__main__":
    # Chỉ khởi động luồng trong process chính, không trong reload
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        start_background_task()
    socketio.run(app, debug=True)