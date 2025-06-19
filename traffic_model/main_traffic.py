# /1_traffic_model/main_traffic.py

import numpy as np
import logging
from .trip_generator import TripGenerator
from .vehicle_movement import VehicleMovement

logger = logging.getLogger(__name__)

class TrafficModel:
    """
    Main traffic model for simulating vehicle movements and EV distribution.
    This class orchestrates trip generation and vehicle movement.
    """
    
    def __init__(self, config, data_loader):
        self.config = config
        self.data_loader = data_loader
        self.vehicles = []
        self.road_network = self.data_loader.load_road_network()
        
        self.trip_generator = TripGenerator(self.config, self.data_loader)
        self.vehicle_movement = VehicleMovement(self.road_network['segments'], self.config)
        
        self.daily_trips = {} # Cache for daily trip patterns
        
        self._initialize_vehicles()

    def _initialize_vehicles(self):
        """Initialize the vehicle population with EVs."""
        total_vehicles = self.config.traffic_params['total_vehicles']
        ev_count = int(total_vehicles * self.config.traffic_params['ev_penetration'])
        
        self.vehicles = []
        ev_types = self.data_loader.load_ev_registration_data()
        
        for i in range(total_vehicles):
            is_ev = i < ev_count
            vehicle = {
                'id': i,
                'type': 'EV' if is_ev else 'ICE',
                'is_bdwpt_equipped': False, # Set later by scenario
                'battery_capacity_kwh': 0,
                'current_soc': 0,
                'location': 'home', # Start at home
                'status': 'parked',
            }
            
            if is_ev:
                # Assign a random EV type based on registration stats
                ev_type = ev_types.sample(n=1, weights='count').iloc[0]
                vehicle['battery_capacity_kwh'] = ev_type['battery_capacity_kwh']
                vehicle['current_soc'] = np.clip(
                    np.random.normal(
                        self.config.ev_params['initial_soc_mean'],
                        self.config.ev_params['initial_soc_std']
                    ), 0.1, 1.0)
            
            self.vehicles.append(vehicle)
        
        logger.info(f"Initialized {total_vehicles} vehicles ({ev_count} EVs).")

    def set_bdwpt_penetration(self, penetration_percent):
        """Set BDWPT equipment penetration for the EV fleet."""
        ev_indices = [i for i, v in enumerate(self.vehicles) if v['type'] == 'EV']
        num_bdwpt = int(len(ev_indices) * penetration_percent / 100)
        
        # Reset all first
        for v in self.vehicles:
            v['is_bdwpt_equipped'] = False

        # Randomly select EVs to equip
        if num_bdwpt > 0 and len(ev_indices) > 0:
            equipped_indices = np.random.choice(ev_indices, num_bdwpt, replace=False)
            for i in equipped_indices:
                self.vehicles[i]['is_bdwpt_equipped'] = True
        
        logger.info(f"Set BDWPT penetration to {penetration_percent}% ({num_bdwpt} equipped vehicles).")

    def get_daily_trip_pattern(self, day_type):
        """Generate or retrieve from cache the trip patterns for a given day type."""
        if day_type not in self.daily_trips:
            logger.info(f"Generating new trip patterns for {day_type}...")
            self.daily_trips[day_type] = self.trip_generator.generate_daily_trips(
                len(self.vehicles), day_type
            )
        return self.daily_trips[day_type]

    def update_vehicle_positions(self, current_time_minutes, day_type):
        """Update vehicle positions for the current time step."""
        trips_df = self.get_daily_trip_pattern(day_type)
        return self.vehicle_movement.update_positions(self.vehicles, trips_df, current_time_minutes)
    
    def generate_trip_patterns(self, hour, day_type):
        """Generate trip patterns for the given hour and day type."""
        # This method is called by the simulation engine
        # For now, we'll just return the cached daily patterns
        return self.get_daily_trip_pattern(day_type)

    def get_bdwpt_vehicles_by_node(self, power_node):
        """Get BDWPT-equipped vehicles currently at a specific power grid node."""
        vehicles_at_node = [
            v for v in self.vehicles
            if v.get('is_bdwpt_equipped') and v.get('location') == power_node
        ]
        return vehicles_at_node