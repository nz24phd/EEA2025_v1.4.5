# 1_traffic_model/data_loader.py - Data loading utilities for traffic model

import os
import numpy as np
import pandas as pd
import json
import logging

logger = logging.getLogger(__name__)

class TrafficDataLoader:
    """Load and preprocess traffic data for simulation"""
    
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.traffic_data = None
        self.census_data = None
        self.road_network = None
        
    def load_traffic_patterns(self, filename="wellington_traffic_patterns.csv"):
        """Load hourly traffic patterns"""
        filepath = os.path.join(self.data_dir, filename)
        
        # If file doesn't exist, create synthetic data
        if not os.path.exists(filepath):
            logger.info("Traffic pattern file not found, generating synthetic data...")
            self.traffic_data = self._generate_synthetic_traffic_patterns()
            # Save for future use
            self.traffic_data.to_csv(filepath, index=False)
        else:
            self.traffic_data = pd.read_csv(filepath)
            
        logger.info(f"Loaded traffic patterns: {self.traffic_data.shape}")
        return self.traffic_data
        
    def load_census_data(self, filename="wellington_census.csv"):
        """Load census/demographic data for trip generation"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.info("Census data not found, generating synthetic data...")
            self.census_data = self._generate_synthetic_census_data()
            self.census_data.to_csv(filepath, index=False)
        else:
            self.census_data = pd.read_csv(filepath)
            
        logger.info(f"Loaded census data: {self.census_data.shape}")
        return self.census_data
        
    def load_road_network(self, filename="wellington_roads.json"):
        """Load road network topology"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            logger.info("Road network not found, generating synthetic network...")
            self.road_network = self._generate_synthetic_road_network()
            with open(filepath, 'w') as f:
                json.dump(self.road_network, f, indent=2)
        else:
            with open(filepath, 'r') as f:
                self.road_network = json.load(f)
                
        logger.info(f"Loaded road network with {len(self.road_network['segments'])} segments")
        return self.road_network
        
    def load_ev_registration_data(self, filename="ev_registrations.csv"):
        """Load EV registration statistics"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            # Create synthetic EV data based on NZ statistics
            ev_data = pd.DataFrame({
                'vehicle_type': ['Nissan Leaf', 'Tesla Model 3', 'MG ZS EV', 'Hyundai Kona', 'Other'],
                'count': [3500, 2000, 1500, 1000, 2000],
                'battery_capacity_kwh': [40, 75, 44, 64, 60],
                'typical_range_km': [270, 500, 320, 450, 400]
            })
            ev_data.to_csv(filepath, index=False)
            return ev_data
        else:
            return pd.read_csv(filepath)
            
    def _generate_synthetic_traffic_patterns(self):
        """Generate synthetic hourly traffic patterns for Wellington"""
        hours = np.arange(24)
        
        # Weekday pattern
        weekday_pattern = np.array([
            0.3, 0.25, 0.2, 0.2, 0.25, 0.4, 0.7, 0.95,  # 0-7 (morning peak)
            1.0, 0.8, 0.7, 0.65, 0.7, 0.7, 0.75, 0.8,   # 8-15 (midday)
            0.9, 0.95, 0.85, 0.7, 0.6, 0.5, 0.4, 0.35   # 16-23 (evening)
        ])
        
        # Weekend pattern
        weekend_pattern = np.array([
            0.25, 0.2, 0.15, 0.15, 0.15, 0.2, 0.3, 0.45,  # 0-7
            0.6, 0.75, 0.85, 0.9, 0.95, 0.9, 0.85, 0.8,   # 8-15
            0.75, 0.7, 0.65, 0.6, 0.5, 0.4, 0.35, 0.3     # 16-23
        ])
        
        # Add some noise
        weekday_pattern += np.random.normal(0, 0.05, 24)
        weekend_pattern += np.random.normal(0, 0.05, 24)
        
        # Normalize
        weekday_pattern = np.clip(weekday_pattern, 0.1, 1.0)
        weekend_pattern = np.clip(weekend_pattern, 0.1, 1.0)
        
        df = pd.DataFrame({
            'hour': hours,
            'weekday_factor': weekday_pattern,
            'weekend_factor': weekend_pattern
        })
        
        return df
        
    def _generate_synthetic_census_data(self):
        """Generate synthetic census data for Wellington suburbs"""
        suburbs = ['CBD', 'Karori', 'Kelburn', 'Newtown', 'Miramar', 
                  'Johnsonville', 'Khandallah', 'Island Bay', 'Lyall Bay', 'Kilbirnie']
        
        data = []
        for suburb in suburbs:
            data.append({
                'suburb': suburb,
                'population': np.random.randint(5000, 20000),
                'households': np.random.randint(2000, 8000),
                'vehicles_per_household': np.random.uniform(1.2, 2.0),
                'employment_rate': np.random.uniform(0.6, 0.8),
                'avg_income': np.random.randint(50000, 120000),
                'distance_to_cbd_km': np.random.uniform(1, 15)
            })
            
        return pd.DataFrame(data)
        
    def _generate_synthetic_road_network(self):
        """Generate synthetic road network matching power grid nodes"""
        # Create road segments that align with IEEE 13-bus nodes
        segments = []
        
        # Main arterial roads
        main_roads = [
            {'id': 'road_650_632', 'from_node': 650, 'to_node': 632, 'length_km': 2.5, 'type': 'arterial'},
            {'id': 'road_632_671', 'from_node': 632, 'to_node': 671, 'length_km': 1.8, 'type': 'arterial'},
            {'id': 'road_671_680', 'from_node': 671, 'to_node': 680, 'length_km': 1.2, 'type': 'arterial'},
            {'id': 'road_632_633', 'from_node': 632, 'to_node': 633, 'length_km': 0.8, 'type': 'collector'},
            {'id': 'road_633_634', 'from_node': 633, 'to_node': 634, 'length_km': 0.5, 'type': 'local'},
        ]
        
        # Secondary roads
        secondary_roads = [
            {'id': 'road_632_645', 'from_node': 632, 'to_node': 645, 'length_km': 0.7, 'type': 'collector'},
            {'id': 'road_645_646', 'from_node': 645, 'to_node': 646, 'length_km': 0.4, 'type': 'local'},
            {'id': 'road_671_692', 'from_node': 671, 'to_node': 692, 'length_km': 1.0, 'type': 'collector'},
            {'id': 'road_692_675', 'from_node': 692, 'to_node': 675, 'length_km': 1.3, 'type': 'collector'},
        ]
        
        segments.extend(main_roads)
        segments.extend(secondary_roads)
        
        # Add traffic capacity based on road type
        for segment in segments:
            if segment['type'] == 'arterial':
                segment['capacity_veh_per_hour'] = np.random.randint(2000, 3000)
            elif segment['type'] == 'collector':
                segment['capacity_veh_per_hour'] = np.random.randint(1000, 1500)
            else:  # local
                segment['capacity_veh_per_hour'] = np.random.randint(500, 800)
                
        return {
            'segments': segments,
            'nodes': list(range(632, 693)),  # Power grid nodes
            'bdwpt_coverage': [632, 633, 634, 645, 646, 671, 675, 680]  # Nodes with BDWPT
        }
        
    def load_trip_distance_distribution(self):
        """Load or generate trip distance distribution"""
        # Based on NZ Household Travel Survey data
        distances_km = np.array([1, 2, 5, 10, 20, 50, 100])
        probabilities = np.array([0.15, 0.25, 0.30, 0.20, 0.08, 0.02])
        
        return {
            'distances': distances_km,
            'probabilities': probabilities,
            'avg_distance': np.sum(distances_km * probabilities)
        }
        
    def load_charging_behavior_data(self):
        """Load EV charging behavior patterns"""
        return {
            'home_charging_probability': 0.85,
            'work_charging_probability': 0.30,
            'public_charging_probability': 0.15,
            'typical_charging_duration_hours': {
                'home': 8,
                'work': 4,
                'public': 0.5
            },
            'preferred_soc_range': (0.5, 0.9),
            'anxiety_soc_threshold': 0.3
        }
        
    def save_processed_data(self, data, filename):
        """Save processed data for future use"""
        filepath = os.path.join(self.data_dir, filename)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
        elif isinstance(data, dict):
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
        else:
            np.save(filepath, data)
            
        logger.info(f"Saved processed data to {filepath}")