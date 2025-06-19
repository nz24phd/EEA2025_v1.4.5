# power_grid_model/bdwpt_agent.py - BDWPT vehicle agent with control logic

import numpy as np
import logging

logger = logging.getLogger(__name__)

class BDWPTAgent:
    """BDWPT-equipped EV agent with intelligent control logic"""
    
    def __init__(self, vehicle_id, battery_capacity, config):
        self.vehicle_id = vehicle_id
        self.battery_capacity = battery_capacity
        self.config = config
        self.soc = 0.7  # Initial state of charge
        self.mode = 'idle'  # idle, G2V, V2G
        self.power_setpoint = 0  # kW
        self.energy_exchanged = 0  # kWh
        self.operation_history = []
        
    def decide_action(self, voltage_pu, tariff, time_step_minutes=1):
        """
        Main decision logic for BDWPT operation
        
        Args:
            voltage_pu: Grid voltage at current location (p.u.)
            tariff: Current electricity price ($/kWh)
            time_step_minutes: Duration of time step
            
        Returns:
            dict: Action containing mode and power setpoint
        """
        # Previous mode for hysteresis
        previous_mode = self.mode
        
        # Get control parameters
        params = self.config.control_params
        
        # Priority 1: SoC-based constraints
        if self.soc < params['soc_force_charge']:
            # Critical low SoC - must charge
            self.mode = 'G2V'
            self.power_setpoint = self.config.bdwpt_params['charging_power_kw']
            logger.debug(f"Vehicle {self.vehicle_id}: Force charging due to low SoC ({self.soc:.2f})")
            
        elif self.soc > params['soc_force_discharge']:
            # Very high SoC - prefer discharging
            self.mode = 'V2G'
            self.power_setpoint = -self.config.bdwpt_params['discharging_power_kw']
            logger.debug(f"Vehicle {self.vehicle_id}: Force discharging due to high SoC ({self.soc:.2f})")
            
        # Priority 2: Grid support based on voltage
        elif voltage_pu > params['voltage_critical_high']:
            # Critical overvoltage - must discharge if possible
            if self.soc > params['soc_min_v2g']:
                self.mode = 'V2G'
                self.power_setpoint = -self.config.bdwpt_params['discharging_power_kw']
                logger.debug(f"Vehicle {self.vehicle_id}: V2G for critical overvoltage ({voltage_pu:.3f} p.u.)")
            else:
                self.mode = 'idle'
                self.power_setpoint = 0
                
        elif voltage_pu < params['voltage_critical_low']:
            # Critical undervoltage - must reduce load or charge less
            if self.mode == 'G2V':
                # Reduce charging power
                self.power_setpoint = self.config.bdwpt_params['charging_power_kw'] * 0.5
            else:
                self.mode = 'idle'
                self.power_setpoint = 0
            logger.debug(f"Vehicle {self.vehicle_id}: Reduced operation for critical undervoltage ({voltage_pu:.3f} p.u.)")
            
        # Priority 3: Economic optimization with grid support
        else:
            # Calculate decision score based on multiple factors
            score = self._calculate_decision_score(voltage_pu, tariff)
            
            if score > 0.2:  # Threshold for V2G
                if self.soc > params['soc_min_v2g']:
                    self.mode = 'V2G'
                    # Modulate power based on score
                    power_factor = min(1.0, score)
                    self.power_setpoint = -self.config.bdwpt_params['discharging_power_kw'] * power_factor
                else:
                    self.mode = 'idle'
                    self.power_setpoint = 0
                    
            elif score < -0.2:  # Threshold for G2V
                if self.soc < self.config.ev_params['max_soc_threshold']:
                    self.mode = 'G2V'
                    # Modulate power based on score
                    power_factor = min(1.0, abs(score))
                    self.power_setpoint = self.config.bdwpt_params['charging_power_kw'] * power_factor
                else:
                    self.mode = 'idle'
                    self.power_setpoint = 0
                    
            else:
                # Near optimal conditions - idle
                self.mode = 'idle'
                self.power_setpoint = 0
                
        # Apply hysteresis to prevent oscillation
        if previous_mode != 'idle' and self.mode != previous_mode:
            # Add small penalty for mode switching
            if abs(self.power_setpoint) < 10:  # kW threshold
                self.mode = previous_mode
                self.power_setpoint = self._get_reduced_power(previous_mode)
                
        # Update SoC based on action
        self._update_soc(time_step_minutes)
        
        # Record decision
        action = {
            'mode': self.mode,
            'power_kw': self.power_setpoint,
            'soc': self.soc,
            'voltage_pu': voltage_pu,
            'tariff': tariff
        }
        
        self.operation_history.append(action)
        
        return action
        
    def _calculate_decision_score(self, voltage_pu, tariff):
        """
        Calculate decision score for V2G/G2V operation
        Positive score favors V2G, negative favors G2V
        """
        params = self.config.control_params
        
        # Voltage score (positive for high voltage, negative for low)
        voltage_score = 0
        if voltage_pu > params['voltage_high_threshold']:
            voltage_score = (voltage_pu - params['voltage_high_threshold']) / 0.05
        elif voltage_pu < params['voltage_low_threshold']:
            voltage_score = (voltage_pu - params['voltage_low_threshold']) / 0.05
              # Tariff score (positive for high tariff, negative for low)
        tariff_score = 0
        if tariff > params['tariff_high_threshold']:
            tariff_score = 1.0
        elif tariff < params['tariff_low_threshold']:
            tariff_score = -1.0
        else:
            # Linear interpolation
            tariff_range = params['tariff_high_threshold'] - params['tariff_low_threshold']
            tariff_score = 2 * (tariff - params['tariff_low_threshold']) / tariff_range - 1
            
        # SoC score (affects willingness to charge/discharge)
        soc_mid = 0.65  # Target SoC
        soc_score = (self.soc - soc_mid) / 0.35
        
        # Combine scores with weights
        total_score = (
            0.4 * voltage_score +
            0.4 * tariff_score +
            0.2 * soc_score
        )
        
        return total_score
        
    def _get_reduced_power(self, mode):
        """Get reduced power for hysteresis"""
        if mode == 'G2V':
            return self.config.bdwpt_params['charging_power_kw'] * 0.7
        elif mode == 'V2G':
            return -self.config.bdwpt_params['discharging_power_kw'] * 0.7
        return 0
        
    def _update_soc(self, time_step_minutes):
        """Update SoC based on power exchange"""
        if self.power_setpoint != 0:
            # Energy exchanged in this time step (kWh)
            energy_kwh = self.power_setpoint * (time_step_minutes / 60)
            
            # Apply efficiency
            if self.power_setpoint > 0:  # Charging
                energy_kwh *= self.config.bdwpt_params['efficiency']
            else:  # Discharging
                energy_kwh /= self.config.bdwpt_params['efficiency']
                
            # Update SoC
            soc_change = energy_kwh / self.battery_capacity
            self.soc = np.clip(self.soc + soc_change, 0.0, 1.0)
            
            # Track total energy exchanged
            self.energy_exchanged += energy_kwh
            
    def update_soc_from_driving(self, distance_km):
        """Update SoC based on driving energy consumption"""
        energy_consumed = distance_km * self.config.ev_params['energy_consumption_kwh_per_km']
        soc_change = energy_consumed / self.battery_capacity
        self.soc = max(0.0, self.soc - soc_change)
        
    def reset_daily(self):
        """Reset daily counters"""
        self.energy_exchanged = 0
        self.operation_history = []
        
    def get_statistics(self):
        """Get operation statistics"""
        if not self.operation_history:
            return {}
            
        stats = {
            'total_energy_charged': sum(h['power_kw'] * (1/60) for h in self.operation_history if h['power_kw'] > 0),
            'total_energy_discharged': abs(sum(h['power_kw'] * (1/60) for h in self.operation_history if h['power_kw'] < 0)),
            'time_charging': sum(1 for h in self.operation_history if h['mode'] == 'G2V'),
            'time_discharging': sum(1 for h in self.operation_history if h['mode'] == 'V2G'),
            'time_idle': sum(1 for h in self.operation_history if h['mode'] == 'idle'),
            'final_soc': self.soc,
            'min_soc': min(h['soc'] for h in self.operation_history),
            'max_soc': max(h['soc'] for h in self.operation_history),
        }
        
        return stats