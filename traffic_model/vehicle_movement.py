# traffic_model/vehicle_movement.py - 修复版本

import numpy as np
import logging

logger = logging.getLogger(__name__)

class VehicleMovement:
    """Handles vehicle movement simulation on the road network."""

    def __init__(self, road_network, config):
        self.road_network = road_network
        self.config = config
        self.vehicle_positions = {}

    def update_positions(self, vehicles, trips, current_time_minutes):
        """
        Updates vehicle positions based on active trips for the current time step.
        """
        # Reset all vehicle statuses before updating
        for v in vehicles:
            if v['status'] == 'driving':
                v['status'] = 'parked'

        active_trips = trips[
            (trips['departure_time'] <= current_time_minutes) &
            (trips['arrival_time'] > current_time_minutes)
        ]

        for _, trip in active_trips.iterrows():
            vehicle_id = trip['vehicle_id']
            destination_node = trip['destination']
            
            if 0 <= vehicle_id < len(vehicles):
                # 确保destination_node是整数类型
                if isinstance(destination_node, str) and destination_node != 'home':
                    try:
                        destination_node = int(destination_node)
                    except ValueError:
                        logger.warning(f"Invalid destination node: {destination_node}, skipping")
                        continue
                elif destination_node == 'home':
                    destination_node = 'home'  # 保持字符串
                
                # Update vehicle's status and location
                vehicles[vehicle_id]['status'] = 'driving'
                vehicles[vehicle_id]['location'] = destination_node
                logger.debug(f"Vehicle {vehicle_id} is active on trip to node {destination_node}")

        # Update status for vehicles that just finished a trip
        finished_trips = trips[trips['arrival_time'] == current_time_minutes]
        for _, trip in finished_trips.iterrows():
            vehicle_id = trip['vehicle_id']
            if 0 <= vehicle_id < len(vehicles):
                destination = trip['destination']
                
                # 确保目的地格式一致
                if isinstance(destination, str) and destination != 'home':
                    try:
                        destination = int(destination)
                    except ValueError:
                        destination = 'home'  # 默认回家
                
                vehicles[vehicle_id]['status'] = 'parked'
                vehicles[vehicle_id]['location'] = destination
                logger.debug(f"Vehicle {vehicle_id} finished trip. Location parked at {destination}")

        # Return empty dict for compatibility
        vehicles_on_segments = {}
        return vehicles_on_segments