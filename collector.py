import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict
import logging

class TomTomTrafficCollector:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.tomtom.com/traffic/services/4"
        
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
        current_speed = data.get('currentSpeed', 0)
        free_flow_speed = data.get('freeFlowSpeed', 1)
        
        if free_flow_speed == 0:
            return 0
            
        return max(0, min(100, (1 - current_speed/free_flow_speed) * 100))

    def get_realtime_data(self, locations: List[Dict]) -> pd.DataFrame:
        current_time = datetime.now()
        data_frames = []
        
        for location in locations:
            try:
                traffic_data = self.get_traffic_flow(location['latitude'], location['longitude'])
                if traffic_data:
                    processed_data = self.process_traffic_data(traffic_data)
                    if processed_data:
                        processed_data.update({
                            'location_name': location['name'],
                            'latitude': location['latitude'],
                            'longitude': location['longitude'],
                            'time_of_day': float(current_time.hour),
                            'day_of_week': current_time.weekday()
                        })
                        df = pd.DataFrame([processed_data])
                        data_frames.append(df)
                        self.logger.info(f"Collected real-time data for {location['name']} at {current_time}")
            except Exception as e:
                self.logger.error(f"Error collecting data for {location['name']}: {str(e)}")
        
        if data_frames:
            return pd.concat(data_frames, ignore_index=True)
        return pd.DataFrame()