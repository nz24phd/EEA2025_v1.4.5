# cosimulation/simulation_engine.py - Main co-simulation engine

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from tqdm import tqdm

from power_grid_model.bdwpt_agent import BDWPTAgent

logger = logging.getLogger(__name__)

class CoSimulationEngine:
    """Co-simulation engine coordinating traffic and power grid models"""
    
    def __init__(self, config, traffic_model, power_grid):
        self.config = config
        self.traffic_model = traffic_model
        self.power_grid = power_grid
        self.bdwpt_agents = {}
        self.results = None
        
    def run_simulation(self, scenario):
        """Run complete co-simulation for a scenario"""
        logger.info(f"Starting co-simulation for scenario: {scenario['name']}")
          # Initialize simulation
        self._initialize_simulation(scenario)
        
        # Prepare results storage
        results_data = []
        
        # Get time series
        time_series = self.config.get_time_series()
        time_steps = time_series['time_steps']
        
        # Main simulation loop
        for t, timestamp in enumerate(tqdm(time_steps, desc="Simulation Progress")):
            # Get hour of day for tariff and load profile
            hour = timestamp.hour
            minute_of_day = timestamp.hour * 60 + timestamp.minute
            
            # Step 1: Update traffic model
            self._update_traffic(timestamp, scenario['day_type'])
            
            # Step 2: Calculate BDWPT power at each node
            bdwpt_powers = self._calculate_bdwpt_powers(hour)
            
            # Step 3: Update power grid loads
            self._update_grid_loads(hour, scenario['load_profile'], bdwpt_powers)
            
            # Step 4: Solve power flow
            pf_results = self.power_grid.solve_power_flow()
            
            # Step 5: Store results
            step_results = self._collect_step_results(
                timestamp, pf_results, bdwpt_powers
            )
            results_data.append(step_results)
            
        # Compile final results
        self.results = self._compile_results(results_data, scenario)
        
        logger.info("Co-simulation completed successfully")
        return self.results
        
    def _initialize_simulation(self, scenario):
        """Initialize simulation components"""
        # Set BDWPT penetration
        self.traffic_model.set_bdwpt_penetration(scenario['bdwpt_penetration'])
          # Create BDWPT agents for equipped vehicles
        self.bdwpt_agents = {}
        for vehicle in self.traffic_model.vehicles:
            if vehicle['is_bdwpt_equipped']:
                agent = BDWPTAgent(
                    vehicle['id'],
                    vehicle['battery_capacity_kwh'],
                    self.config
                )
                # Set initial SoC
                agent.soc = vehicle['current_soc']
                self.bdwpt_agents[vehicle['id']] = agent
                
        logger.info(f"Initialized {len(self.bdwpt_agents)} BDWPT agents")
        
    def _update_traffic(self, timestamp, day_type):
        """Update traffic model for current time step"""
        hour = timestamp.hour
        
        # Generate new trips
        trips = self.traffic_model.generate_trip_patterns(hour, day_type)
          # Update vehicle positions
        minute_of_day = hour * 60 + timestamp.minute
        vehicles_on_roads = self.traffic_model.update_vehicle_positions(minute_of_day, day_type)
        
        # Update SoC for driving vehicles
        for vehicle in self.traffic_model.vehicles:
            if vehicle['status'] == 'driving' and vehicle['id'] in self.bdwpt_agents:
                # Simple energy consumption based on time step
                distance = self.config.traffic_params['average_trip_distance_km'] / 30  # km per minute
                self.bdwpt_agents[vehicle['id']].update_soc_from_driving(distance)
                
    def _calculate_bdwpt_powers(self, hour):
        """Calculate BDWPT power exchange at each node"""
        bdwpt_powers = {}
        
        # Get current tariff
        tariff = self.config.get_tariff_at_hour(hour)
        
        # For each BDWPT-enabled node
        for node in self.config.grid_params['bdwpt_nodes']:
            total_power = 0
              # Get vehicles at this node
            vehicles = self.traffic_model.get_bdwpt_vehicles_by_node(node)
            
            if vehicles:
                logger.info(f"Found {len(vehicles)} BDWPT-equipped vehicles at node {node}")

            for vehicle in vehicles:
                if vehicle['id'] in self.bdwpt_agents:
                    agent = self.bdwpt_agents[vehicle['id']]
                    
                    # Get voltage at this node
                    try:
                        voltage = self.power_grid.get_voltage(node)
                        logger.debug(f"Got voltage {voltage} for node {node}")
                    except Exception as e:
                        logger.error(f"Error getting voltage for node {node}: {e}")
                        voltage = 1.0  # Default voltage
                    
                    # Agent decides action
                    try:
                        action = agent.decide_action(voltage, tariff, self.config.time_step_minutes)
                        logger.debug(f"Agent {agent.vehicle_id} action: {action}")
                    except Exception as e:
                        logger.error(f"Error in agent decision for vehicle {vehicle['id']}: {e}")
                        action = {'power_kw': 0}                    # Accumulate power
                    total_power += action['power_kw']
                    
            bdwpt_powers[node] = total_power
            
        logger.debug(f"Calculated BDWPT powers at hour {hour}: {bdwpt_powers}")
        return bdwpt_powers
        
    def _update_grid_loads(self, hour, load_profile_type, bdwpt_powers):
        """Update power grid loads including BDWPT"""
        # Get base load multiplier from profile
        time_minutes = hour * 60
        day_type = 'weekday' if 'Weekday' in load_profile_type else 'weekend'
          # Update base loads
        for bus, load in self.power_grid.loads.items():
            try:
                # Get node-specific load profile
                load_kw = self.config.get_load_profile(bus, time_minutes, day_type)
                logger.debug(f"Load for bus {bus}: {load_kw} kW")
            except Exception as e:
                logger.error(f"Error getting load profile for bus {bus}: {e}")
                logger.error(f"bus type: {type(bus)}, time_minutes type: {type(time_minutes)}, day_type: {day_type}")
                raise e
            
        # FIX: Use the new update_bdwpt_load method
        if any(p != 0 for p in bdwpt_powers.values()):
            logger.info(f"Updating grid with non-zero BDWPT powers: {bdwpt_powers}")
        for node, power in bdwpt_powers.items():
            self.power_grid.update_bdwpt_load(node, power)
            
    def _collect_step_results(self, timestamp, pf_results, bdwpt_powers):
        """Collect results for current time step"""
        results = {
            'timestamp': timestamp,
            'converged': pf_results['converged'],
            'total_load_kw': pf_results['powers']['total_load'],
            'total_losses_kw': pf_results['powers']['total_losses'],
            'total_bdwpt_kw': sum(bdwpt_powers.values()),
            'bdwpt_charging_kw': sum(p for p in bdwpt_powers.values() if p > 0),
            'bdwpt_discharging_kw': abs(sum(p for p in bdwpt_powers.values() if p < 0)),
        }
        
        # Add voltage data
        for bus, voltage in pf_results['voltages'].items():
            results[f'voltage_bus_{bus}'] = voltage
            
        # Add BDWPT power by node
        for node, power in bdwpt_powers.items():
            results[f'bdwpt_node_{node}_kw'] = power
              # Count vehicles in different modes
        mode_counts = {'G2V': 0, 'V2G': 0, 'idle': 0}
        for agent in self.bdwpt_agents.values():
            if agent.operation_history:
                mode_counts[agent.mode] += 1
        results.update({f'vehicles_{mode}': count for mode, count in mode_counts.items()})
        
        return results
    
    def _compile_results(self, results_data, scenario):
        """Compile simulation results into final format"""
        logger.info(f"Compiling results for {len(results_data)} time steps")
        
        # Debug: Check results_data structure
        if results_data:
            sample_keys = list(results_data[0].keys()) if results_data[0] else []
            logger.info(f"Sample step result keys: {sample_keys}")
        
        # Create DataFrame from results
        df = pd.DataFrame(results_data)
        logger.info(f"Created DataFrame with shape: {df.shape}, columns: {list(df.columns)}")
        
        # Get time step minutes with fallback
        time_step_minutes = getattr(self.config, 'time_step_minutes', 
                                   self.config.simulation_params.get('time_step_minutes', 15))
        logger.info(f"Using time_step_minutes: {time_step_minutes}")
        
        # Calculate summary statistics
        summary = {
            'scenario': scenario['name'],
            'base_name': scenario['name'],  # Add base name for KPI calculation
            'bdwpt_penetration': scenario['bdwpt_penetration'],
            'peak_load': df['total_load_kw'].max(),
            'min_load': df['total_load_kw'].min(),
            'avg_load': df['total_load_kw'].mean(),
            'total_energy_kwh': df['total_load_kw'].sum() * time_step_minutes / 60,
            'total_losses_kwh': df['total_losses_kw'].sum() * time_step_minutes / 60,
            'min_voltage': df[[col for col in df.columns if 'voltage_bus' in col]].min().min(),
            'max_voltage': df[[col for col in df.columns if 'voltage_bus' in col]].max().max(),
            'bdwpt_energy_charged_kwh': df['bdwpt_charging_kw'].sum() * time_step_minutes / 60,
            'bdwpt_energy_discharged_kwh': df['bdwpt_discharging_kw'].sum() * time_step_minutes / 60,
        }
        
        # Count voltage violations
        voltage_cols = [col for col in df.columns if 'voltage_bus' in col]
        voltage_violations = 0
        for col in voltage_cols:
            violations = ((df[col] < 0.95) | (df[col] > 1.05)).sum()
            voltage_violations += violations
        summary['voltage_violations'] = voltage_violations
        
        # Count reverse power flow events (when BDWPT discharge > local load)
        df['net_load'] = df['total_load_kw'] - df['total_bdwpt_kw']
        summary['reverse_flow_events'] = (df['net_load'] < 0).sum()
        
        # Agent statistics
        agent_stats = []
        for vehicle_id, agent in self.bdwpt_agents.items():
            stats = agent.get_statistics()
            stats['vehicle_id'] = vehicle_id
            agent_stats.append(stats)
        
        return {
            'timeseries': df,
            'summary': summary,
            'agent_stats': pd.DataFrame(agent_stats) if agent_stats else None
        }