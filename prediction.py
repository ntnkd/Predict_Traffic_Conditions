import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from flask import Flask, request, jsonify, render_template_string
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error

class TrafficPredictor:
    def __init__(self, data_path, model_path=None):
        """Khởi tạo TrafficPredictor với dữ liệu và mô hình (nếu có)."""
        self.data_path = data_path
        self.scaler = MinMaxScaler()
        self.sequence_length = 6
        self.features = ['speed', 'vehicle_density', 'time_of_day', 'day_of_week']
        self.speed_thresholds = {'very_slow': 20, 'slow': 40, 'moderate': 60, 'fast': float('inf')}
        self.density_thresholds = {'low': 30, 'medium': 60, 'high': float('inf')}
        self.original_data = pd.read_csv(data_path)
        self.model = tf.keras.models.load_model(model_path) if model_path and os.path.exists(model_path) else None
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        """Tiền xử lý dữ liệu: đọc CSV, chuẩn hóa, chia thành tập huấn luyện và kiểm tra."""
        df = pd.read_csv(self.data_path)
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
        
        self.X_train = np.concatenate([data['X_train'] for data in self.location_data.values()])
        self.X_test = np.concatenate([data['X_test'] for data in self.location_data.values()])
        self.y_train = np.concatenate([data['y_train'] for data in self.location_data.values()])
        self.y_test = np.concatenate([data['y_test'] for data in self.location_data.values()])

    def build_model(self):
        """Xây dựng mô hình LSTM để dự đoán tốc độ và mật độ xe."""
        self.model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(self.sequence_length, len(self.features))),
            Dropout(0.3),
            LSTM(32),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(2)
        ])
        # Tối ưu hóa learning rate với Adam
        optimizer = Adam(learning_rate=0.0005)
        self.model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

    def train_model(self, epochs=400, batch_size=32):
        """Huấn luyện mô hình LSTM."""
        if not self.model:
            self.build_model()

        # Thêm Early Stopping
        early_stopping = EarlyStopping(
            monitor='val_loss',  # Theo dõi validation loss
            patience=20,         # Dừng nếu không cải thiện sau 20 epoch
            restore_best_weights=True  # Khôi phục trọng số tốt nhất
        )

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],  # Áp dụng Early Stopping
            verbose=1
        )

        self.model.save('traffic_model.keras')
        return history

    def evaluate_model(self):
        """Đánh giá và so sánh mô hình LSTM với Linear Regression."""
        # Dự đoán từ mô hình LSTM
        y_pred_scaled_lstm = self.model.predict(self.X_test, verbose=0)
        
        # Chuyển đổi ngược lại từ dữ liệu chuẩn hóa về giá trị thực
        y_test_full = np.concatenate([self.y_test, np.zeros((len(self.y_test), 2))], axis=1)
        y_pred_full_lstm = np.concatenate([y_pred_scaled_lstm, np.zeros((len(y_pred_scaled_lstm), 2))], axis=1)
        
        y_test_real = self.scaler.inverse_transform(y_test_full)[:, :2]
        y_pred_real_lstm = self.scaler.inverse_transform(y_pred_full_lstm)[:, :2]

        # Chuẩn bị dữ liệu cho Linear Regression (chuyển từ 3D sang 2D)
        X_test_2d = self.X_test.reshape(self.X_test.shape[0], -1)  # (samples, sequence_length * features)
        y_test_speed = y_test_real[:, 0]  # Tốc độ
        y_test_density = y_test_real[:, 1]  # Mật độ

        # Huấn luyện và dự đoán với Linear Regression
        lr_speed = LinearRegression()
        lr_density = LinearRegression()
        
        lr_speed.fit(X_test_2d, y_test_speed)
        lr_density.fit(X_test_2d, y_test_density)
        
        y_pred_speed_lr = lr_speed.predict(X_test_2d)
        y_pred_density_lr = lr_density.predict(X_test_2d)

        # Tính toán các chỉ số đánh giá cho LSTM
        mae_speed_lstm = mean_absolute_error(y_test_speed, y_pred_real_lstm[:, 0])
        mse_speed_lstm = mean_squared_error(y_test_speed, y_pred_real_lstm[:, 0])
        rmse_speed_lstm = np.sqrt(mse_speed_lstm)
        
        mae_density_lstm = mean_absolute_error(y_test_density, y_pred_real_lstm[:, 1])
        mse_density_lstm = mean_squared_error(y_test_density, y_pred_real_lstm[:, 1])
        rmse_density_lstm = np.sqrt(mse_density_lstm)

        # Tính toán các chỉ số đánh giá cho Linear Regression
        mae_speed_lr = mean_absolute_error(y_test_speed, y_pred_speed_lr)
        mse_speed_lr = mean_squared_error(y_test_speed, y_pred_speed_lr)
        rmse_speed_lr = np.sqrt(mse_speed_lr)
        
        mae_density_lr = mean_absolute_error(y_test_density, y_pred_density_lr)
        mse_density_lr = mean_squared_error(y_test_density, y_pred_density_lr)
        rmse_density_lr = np.sqrt(mse_density_lr)

        # Tạo bảng so sánh
        comparison_data = {
            'Metric': ['MAE', 'MSE', 'RMSE'],
            'LSTM_Speed': [mae_speed_lstm, mse_speed_lstm, rmse_speed_lstm],
            'LR_Speed': [mae_speed_lr, mse_speed_lr, rmse_speed_lr],
            'LSTM_Density': [mae_density_lstm, mse_density_lstm, rmse_density_lstm],
            'LR_Density': [mae_density_lr, mse_density_lr, rmse_density_lr]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(2)  # Làm tròn đến 2 chữ số thập phân
        
        # In bảng so sánh
        print("\nModel Comparison Table:")
        print(comparison_df.to_string(index=False))
        
        # Trả về kết quả đánh giá dưới dạng dictionary nếu cần
        evaluation_results = {
            'LSTM': {
                'speed': {'MAE': mae_speed_lstm, 'MSE': mse_speed_lstm, 'RMSE': rmse_speed_lstm},
                'density': {'MAE': mae_density_lstm, 'MSE': mse_density_lstm, 'RMSE': rmse_density_lstm}
            },
            'LinearRegression': {
                'speed': {'MAE': mae_speed_lr, 'MSE': mse_speed_lr, 'RMSE': rmse_speed_lr},
                'density': {'MAE': mae_density_lr, 'MSE': mse_density_lr, 'RMSE': rmse_density_lr}
            }
        }
        
        return evaluation_results

    def predict_future(self, location, minutes_ahead=10):
        """Dự đoán tốc độ và mật độ xe trong tương lai cho một vị trí."""
        location_data = self.location_data[location]
        latest_sequence = location_data['raw_data'].iloc[-self.sequence_length:][self.features]
        
        # Đảm bảo latest_sequence là DataFrame với tên cột
        latest_sequence_df = pd.DataFrame(latest_sequence, columns=self.features)
        scaled_sequence = self.scaler.transform(latest_sequence_df)
        
        steps_per_5min = 1
        prediction_steps = int(minutes_ahead / 5) * steps_per_5min
        
        current_input = scaled_sequence[np.newaxis, :, :]
        predictions = []
        
        # Lấy thời gian ban đầu để cập nhật
        last_time = pd.to_datetime(location_data['raw_data'].iloc[-1]['timestamp'])
        
        for step in range(prediction_steps):
            pred = self.model.predict(current_input, verbose=0)
            predictions.append(pred[0])
            
            # Tạo đầu vào mới
            new_input = np.zeros((1, self.sequence_length, len(self.features)))
            new_input[0, :-1] = current_input[0, 1:]
            
            # Đảm bảo pred trong khoảng hợp lý
            pred_adjusted = np.clip(pred[0], 0, 1)
            
            # Cập nhật tốc độ và mật độ
            new_input[0, -1, :2] = pred_adjusted
            
            # Cập nhật thời gian
            new_time = last_time + pd.Timedelta(minutes=5 * (step + 1))
            new_time_of_day = new_time.hour + new_time.minute / 60
            new_day_of_week = new_time.weekday()
            
            # Tạo DataFrame cho thời gian mới để transform
            time_features_df = pd.DataFrame([[0, 0, new_time_of_day, new_day_of_week]], 
                                        columns=self.features)
            time_features = self.scaler.transform(time_features_df)[0, 2:]
            new_input[0, -1, 2:] = time_features
            
            current_input = new_input
        
        predictions = np.array(predictions)
        # Chuyển predictions thành DataFrame trước khi inverse_transform
        full_predictions_df = pd.DataFrame(np.concatenate([predictions, np.zeros((len(predictions), 2))], axis=1),
                                        columns=self.features)
        predictions_inv = self.scaler.inverse_transform(full_predictions_df)[:, :2]
        
        # Đảm bảo mật độ trong khoảng 0-100%
        predictions_inv[:, 1] = np.clip(predictions_inv[:, 1], 0, 100)
        
        return predictions_inv[-1]

    def _get_speed_color(self, speed):
        """Xác định màu sắc dựa trên tốc độ."""
        if speed <= self.speed_thresholds['very_slow']:
            return 'red'
        elif speed <= self.speed_thresholds['slow']:
            return 'orange'
        elif speed <= self.speed_thresholds['moderate']:
            return 'yellow'
        else:
            return 'green'

    def _get_density_color(self, density):
        """Xác định màu sắc dựa trên mật độ xe."""
        if density <= self.density_thresholds['low']:
            return 'green'
        elif density <= self.density_thresholds['medium']:
            return 'yellow'
        else:
            return 'red'

    def _get_speed_category(self, speed):
        """Xác định danh mục tốc độ."""
        if speed <= self.speed_thresholds['very_slow']:
            return 'Very Slow'
        elif speed <= self.speed_thresholds['slow']:
            return 'Slow'
        elif speed <= self.speed_thresholds['moderate']:
            return 'Moderate'
        else:
            return 'Fast'

    def _get_density_category(self, density):
        """Xác định danh mục mật độ xe."""
        if density <= self.density_thresholds['low']:
            return 'Low'
        elif density <= self.density_thresholds['medium']:
            return 'Medium'
        else:
            return 'High'

    def create_traffic_map_data(self, minutes_ahead=10):
        """Tạo dữ liệu bản đồ giao thông cho Leaflet."""
        unique_locations = self.original_data.groupby('location_name').agg({
            'latitude': 'first',
            'longitude': 'first'
        }).reset_index()
        
        map_data = []
        for _, row in unique_locations.iterrows():
            location = row['location_name']
            pred = self.predict_future(location, minutes_ahead)
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

def initialize_predictor():
    data_path = 'temp_traffic_data.csv'
    model_path = 'traffic_model.keras'
    
    if not os.path.exists(model_path):
        print(f"File mô hình '{model_path}' không tồn tại. Bắt đầu huấn luyện mô hình...")
        predictor = TrafficPredictor(data_path)
        predictor.train_model(epochs=400, batch_size=32)
        print(f"Mô hình đã được huấn luyện và lưu thành '{model_path}'.")
    else:
        print(f"Tìm thấy file mô hình '{model_path}'. Đang nạp mô hình...")
    
    predictor = TrafficPredictor(data_path, model_path)
    # Gọi hàm đánh giá để so sánh
    predictor.evaluate_model()
    return predictor

predictor = initialize_predictor()

@app.route('/')
def index():
    """Trả về trang HTML chính với mã hóa UTF-8."""
    with open('traffic_predictions_map.html', encoding='utf-8') as f:
        return render_template_string(f.read())

@app.route('/predict', methods=['POST'])
def predict():
    """Xử lý yêu cầu dự đoán từ client, trả về dữ liệu bản đồ."""
    minutes_ahead = int(request.json.get('minutes', 10))
    map_data = predictor.create_traffic_map_data(minutes_ahead)
    return jsonify(map_data)

if __name__ == "__main__":
    app.run(debug=True)