# 1_traffic_model/trip_generator.py - Trip generation module

import numpy as np
import pandas as pd
from scipy import stats
import logging

logger = logging.getLogger(__name__)

class TripGenerator:
    """Generate realistic trip patterns for vehicles based on activity-based modeling"""
    
    def __init__(self, config, data_loader=None):
        self.config = config
        self.data_loader = data_loader
        self.road_network = self.data_loader.load_road_network()
        self.nodes = self.road_network['nodes']
        
        # Trip purpose distribution (based on NZ Household Travel Survey)
        self.trip_purposes = {
            'home': 0.25,
            'work': 0.30,
            'shopping': 0.15,
            'education': 0.10,
            'social': 0.10,
            'other': 0.10
        }
        
        # Time-of-day distributions for different trip purposes
        self.departure_distributions = self._initialize_departure_distributions()
        
    def _initialize_departure_distributions(self):
        """Initialize probability distributions for trip departure times"""
        distributions = {}
        
        # Work trips - morning peak
        distributions['work'] = {
            'morning': stats.norm(loc=7.5, scale=1.0),  # Peak at 7:30 AM
            'evening': stats.norm(loc=17.5, scale=1.0)  # Peak at 5:30 PM
        }
        
        # Shopping trips - spread throughout day
        distributions['shopping'] = {
            'distribution': stats.uniform(loc=9, scale=10)  # 9 AM to 7 PM
        }
        
        # Education trips - school hours
        distributions['education'] = {
            'morning': stats.norm(loc=8.0, scale=0.5),   # Peak at 8:00 AM
            'afternoon': stats.norm(loc=15.0, scale=0.5)  # Peak at 3:00 PM
        }
        
        # Social/recreational trips - evening and weekend
        distributions['social'] = {
            'distribution': stats.norm(loc=19.0, scale=2.0)  # Peak at 7 PM
        }
        
        return distributions
        
    def generate_daily_trips(self, num_vehicles, day_type='weekday'):
        """Generate complete daily trip chains for all vehicles"""
        all_trips = []
        
        for vehicle_id in range(num_vehicles):
            vehicle_trips = self.generate_vehicle_trip_chain(
                vehicle_id, day_type
            )
            all_trips.extend(vehicle_trips)
            
        # Convert to DataFrame
        trips_df = pd.DataFrame(all_trips)
        logger.info(f"Generated {len(trips_df)} trips for {num_vehicles} vehicles")
        
        return trips_df
        
    def generate_vehicle_trip_chain(self, vehicle_id, day_type='weekday'):
        """Generate a realistic trip chain for a single vehicle"""
        trips = []
        
        # Determine number of trips (Poisson distribution)
        avg_trips = self.config.traffic_params['trips_per_vehicle_per_day']
        if day_type == 'weekend':
            avg_trips *= 0.7  # Fewer trips on weekends
            
        num_trips = np.random.poisson(avg_trips)
        num_trips = np.clip(num_trips, 0, 8)  # Reasonable bounds
        
        if num_trips == 0:
            return trips
            
        # Generate trip chain
        current_location = 'home'
        current_time = 0  # Minutes since midnight
        locations_visited = []
        
        for trip_num in range(num_trips):
            # Select trip purpose based on context
            purpose = self._select_trip_purpose(
                current_location, 
                locations_visited, 
                day_type,
                current_time
            )
            
            # Determine destination
            destination = self._select_destination(purpose, current_location)
            
            # Generate departure time
            departure_time = self._generate_departure_time(
                purpose, 
                current_time, 
                day_type
            )
            
            # Generate trip distance
            distance_km = self._generate_trip_distance(purpose, destination)
            
            # Calculate travel time (average speed varies by time of day)
            avg_speed_kmh = self._get_average_speed(departure_time)
            duration_minutes = (distance_km / avg_speed_kmh) * 60
            
            # Create trip record
            trip = {
                'vehicle_id': vehicle_id,
                'trip_id': f"{vehicle_id}_{trip_num}",
                'purpose': purpose,
                'origin': current_location,
                'destination': destination,
                'departure_time': departure_time,
                'arrival_time': departure_time + duration_minutes,
                'distance_km': distance_km,
                'duration_minutes': duration_minutes,
                'day_type': day_type
            }
            
            trips.append(trip)
            
            # Update state
            current_location = destination
            current_time = departure_time + duration_minutes
            locations_visited.append(destination)
            
        # Ensure last trip returns home if not already
        if current_location != 'home' and current_time < 23 * 60:
            return_trip = self._generate_return_home_trip(
                vehicle_id, 
                len(trips), 
                current_location, 
                current_time
            )
            trips.append(return_trip)
            
        return trips
        
    def _select_trip_purpose(self, current_location, locations_visited, 
                           day_type, current_time_minutes):
        """Select trip purpose based on context"""
        hour = current_time_minutes / 60
        
        # Context-based rules
        if current_location == 'home':
            if 6 <= hour <= 9 and 'work' not in locations_visited:
                return 'work' if day_type == 'weekday' else 'social'
            elif 7 <= hour <= 9 and 'education' not in locations_visited:
                if np.random.random() < 0.3:  # 30% have education trips
                    return 'education'
                    
        elif current_location == 'work':
            if 11.5 <= hour <= 13.5:
                return 'other'  # Lunch
            elif hour >= 16:
                return 'home'
                
        # Random selection based on general distribution
        purposes = list(self.trip_purposes.keys())
        probs = list(self.trip_purposes.values())
        return np.random.choice(purposes, p=probs)
        
    def _select_destination(self, purpose, current_location):
        """Select destination based on trip purpose using network nodes."""
        if purpose == 'home':
            return 'home'

        # Select a random node from the network, excluding the current location
        possible_destinations = [n for n in self.nodes if n != current_location]
        if not possible_destinations:
             return current_location
        return np.random.choice(possible_destinations)
            
    def _generate_departure_time(self, purpose, earliest_time, day_type):
        """Generate realistic departure time based on purpose"""
        
        if purpose == 'work' and day_type == 'weekday':
            # Morning work trip
            if earliest_time < 9 * 60:
                time_hour = self.departure_distributions['work']['morning'].rvs()
                time_hour = np.clip(time_hour, 6, 10)
            else:
                # Evening return
                time_hour = self.departure_distributions['work']['evening'].rvs()
                time_hour = np.clip(time_hour, 16, 20)
                
        elif purpose == 'education':
            if earliest_time < 12 * 60:
                time_hour = self.departure_distributions['education']['morning'].rvs()
            else:
                time_hour = self.departure_distributions['education']['afternoon'].rvs()
                
        elif purpose == 'shopping':
            time_hour = self.departure_distributions['shopping']['distribution'].rvs()
            
        elif purpose == 'social':
            time_hour = self.departure_distributions['social']['distribution'].rvs()
            if day_type == 'weekend':
                time_hour -= 2  # Earlier social activities on weekends
                
        else:
            # Other trips - uniform throughout the day
            time_hour = np.random.uniform(earliest_time / 60 + 0.5, 22)
            
        # Convert to minutes and ensure it's after earliest time
        departure_time = time_hour * 60
        departure_time = max(departure_time, earliest_time + 15)  # At least 15 min after arrival
        
        return departure_time
        
    def _generate_trip_distance(self, purpose, destination):
        """Generate trip distance based on purpose and destination"""
        # Distance distributions by purpose (log-normal)
        distance_params = {
            'work': (2.5, 0.5),      # mean=12km after exp
            'shopping': (1.8, 0.4),   # mean=6km after exp
            'education': (2.1, 0.4),  # mean=8km after exp
            'social': (2.3, 0.6),     # mean=10km after exp
            'other': (2.0, 0.5),      # mean=7km after exp
            'home': (2.2, 0.5)        # varies
        }
        
        mean_log, std_log = distance_params.get(purpose, (2.0, 0.5))
        distance = np.random.lognormal(mean_log, std_log)
        
        # Clip to reasonable bounds
        distance = np.clip(distance, 0.5, 50)
        
        return distance
        
    def _get_average_speed(self, time_minutes):
        """Get average travel speed based on time of day (congestion)"""
        hour = time_minutes / 60
        
        # Speed profile (km/h) - lower during peak hours
        if 7 <= hour <= 9 or 17 <= hour <= 19:
            # Peak hours - heavy congestion
            return np.random.uniform(20, 30)
        elif 9 <= hour <= 16:
            # Daytime - moderate traffic
            return np.random.uniform(30, 40)
        else:
            # Off-peak - free flow
            return np.random.uniform(40, 50)
            
    def _generate_return_home_trip(self, vehicle_id, trip_num, 
                                  current_location, current_time):
        """Generate a return home trip"""
        distance = self._generate_trip_distance('home', 'home')
        avg_speed = self._get_average_speed(current_time)
        duration = (distance / avg_speed) * 60
        
        return {
            'vehicle_id': vehicle_id,
            'trip_id': f"{vehicle_id}_{trip_num}",
            'purpose': 'home',
            'origin': current_location,
            'destination': 'home',
            'departure_time': current_time + 10,  # 10 min after arrival
            'arrival_time': current_time + 10 + duration,
            'distance_km': distance,
            'duration_minutes': duration,
            'day_type': 'return'
        }
        
    def generate_od_matrix(self, zones, time_period='am_peak'):
        """Generate origin-destination matrix for given zones"""
        n_zones = len(zones)
        od_matrix = np.zeros((n_zones, n_zones))
        
        # Generate trips between zones
        for i, origin in enumerate(zones):
            for j, destination in enumerate(zones):
                if i != j:
                    # Distance-decay function
                    distance = np.abs(i - j) * 2  # Simple distance proxy
                    trips = 100 * np.exp(-0.3 * distance)
                    od_matrix[i, j] = np.random.poisson(trips)
                    
        return pd.DataFrame(od_matrix, index=zones, columns=zones)