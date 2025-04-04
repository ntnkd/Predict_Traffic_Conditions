import requests
import pandas as pd
import time
from datetime import datetime
import json
from typing import List, Dict
import logging
import os

class TomTomTrafficCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tomtom.com/traffic/services/4"
        
        # Thiết lập logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('traffic_collection.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def get_traffic_flow(self, latitude: float, longitude: float, radius: int = 100) -> Dict:
        """
        Lấy dữ liệu lưu lượng giao thông cho một vị trí cụ thể
        """
        endpoint = f"{self.base_url}/flowSegmentData/relative/10/json"
        
        params = {
            'key': self.api_key,
            'point': f"{latitude},{longitude}",
            'radius': radius
        }
        
        try:
            response = requests.get(endpoint, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Error fetching traffic data: {str(e)}")
            return None

    def process_traffic_data(self, raw_data: Dict) -> Dict:
        """
        Xử lý dữ liệu thô từ API
        """
        if not raw_data or 'flowSegmentData' not in raw_data:
            return None
            
        data = raw_data['flowSegmentData']
        
        return {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'speed': data.get('currentSpeed', 0),
            'road_closure': data.get('roadClosure', False),
            'vehicle_density': self._calculate_density(data)
        }

    def _calculate_density(self, data: Dict) -> float:
        """
        Tính toán mật độ phương tiện dựa trên tốc độ hiện tại và tốc độ tự do
        """
        current_speed = data.get('currentSpeed', 0)
        free_flow_speed = data.get('freeFlowSpeed', 1)  # Tránh chia cho 0
        
        if free_flow_speed == 0:
            return 0
            
        return max(0, min(100, (1 - current_speed/free_flow_speed) * 100))

    def collect_and_save_data(self, 
                            locations: List[Dict], 
                            interval: int = 300,
                            duration_hours: float = 24.0):
        """
        Thu thập và lưu dữ liệu cho nhiều vị trí trong một khoảng thời gian
        
        Args:
            locations: List of dicts with 'name', 'latitude', 'longitude'
            interval: Seconds between collections
            duration_hours: How many hours to collect data
        """
        
        total_iterations = int((duration_hours * 3600) / interval)
        data_frames = []
        
        for i in range(total_iterations):
            current_time = datetime.now()
            
            for location in locations:
                try:
                    # Lấy dữ liệu giao thông
                    traffic_data = self.get_traffic_flow(
                        location['latitude'], 
                        location['longitude']
                    )
                    
                    if traffic_data:
                        processed_data = self.process_traffic_data(traffic_data)
                        
                        if processed_data:
                            # Thêm thông tin vị trí và thời gian
                            processed_data.update({
                                'location_name': location['name'],
                                'latitude': location['latitude'],
                                'longitude': location['longitude'],
                                'time_of_day': current_time.hour,
                                'day_of_week': current_time.weekday()
                            })
                            
                            # Tạo DataFrame và thêm vào list
                            df = pd.DataFrame([processed_data])
                            data_frames.append(df)
                            
                            # Lưu dữ liệu tạm thời
                            self._save_temporary_data(df)
                            
                            self.logger.info(
                                f"Collected data for {location['name']} at {current_time}"
                            )
                            
                except Exception as e:
                    self.logger.error(
                        f"Error collecting data for {location['name']}: {str(e)}"
                    )
            
            # Chờ đến lần thu thập tiếp theo
            time.sleep(interval)
        
        # Kết hợp tất cả dữ liệu và lưu
        if data_frames:
            final_df = pd.concat(data_frames, ignore_index=True)
            self._save_final_data(final_df)
            
    def _save_temporary_data(self, df: pd.DataFrame):
        """Lưu dữ liệu tạm thời"""
        temp_file = 'temp_traffic_data.csv'
        mode = 'a' if os.path.exists(temp_file) else 'w'
        df.to_csv(temp_file, mode=mode, header=(mode=='w'), index=False)
        
    def _save_final_data(self, df: pd.DataFrame):
        """Lưu dữ liệu cuối cùng"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'traffic_data_{timestamp}.csv'
        df.to_csv(filename, index=False)
        self.logger.info(f"Saved final data to {filename}")

# Sử dụng class để thu thập dữ liệu
if __name__ == "__main__":
    # Khởi tạo collector với API key của bạn
    api_key = "8pAHAqpTkhGyvgvl1aWLq4nkbYGuM5Ld"
    collector = TomTomTrafficCollector(api_key)
    
    # Định nghĩa các vị trí cần thu thập dữ liệu
    locations = [
        
        {
            'name': 'Dragon Bridge',
            'latitude': 16.061025,
            'longitude': 108.226179 


        },
        {
            'name': 'Han River Bridge',
            'latitude': 16.072118,
            'longitude': 108.22656

        },
        {
            'name': 'Hue T-junction',
            'latitude': 16.060478,
            'longitude': 108.177008
        },
        {
            'name': 'Da Nang station',
            'latitude': 16.070959,
            'longitude': 108.209465
        },
    ]
    
    # Thu thập dữ liệu mỗi 5 phút trong 24 giờ
    collector.collect_and_save_data(
        locations=locations,
        interval=300,  # 5 phút
        duration_hours=24.0
    )