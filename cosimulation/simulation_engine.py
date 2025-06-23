# cosimulation/simulation_engine.py - 最终修复版本

import pandas as pd
import logging
from datetime import timedelta
import os

# 导入模型
from traffic_model.data_loader import TrafficDataLoader
from traffic_model.main_traffic import TrafficModel
from power_grid_model.ieee_13_bus_model import IEEE13BusModel
from power_grid_model.bdwpt_efficiency_model import DynamicBDWPTEfficiency, BDWPTCharacteristics, ThermalState
from .results_analyzer import ResultsAnalyzer, ALL_TIMESERIES_COLUMNS

class SimulationEngine:
    """Orchestrates the co-simulation for a single scenario."""

    def __init__(self, base_config, scenario_config):
        """
        Initializes the simulation engine for a specific scenario.
        
        Args:
            base_config: The main SimulationConfig object.
            scenario_config (dict): The configuration for the specific scenario to run.
        """
        self.base_config = base_config
        self.scenario_config = scenario_config
        self.scenario_name = scenario_config['name']
        
        # --- ▼▼▼ 这里是修改的核心 ▼▼▼ ---

        # 1. 结合主配置和场景配置，确定本次仿真具体的参数
        self.start_time = pd.to_datetime(base_config.simulation_params['start_time'])
        self.end_time = pd.to_datetime(base_config.simulation_params['end_time'])
        
        time_step_in_minutes = base_config.simulation_params['time_step_minutes']
        self.time_step = timedelta(seconds=time_step_in_minutes * 60)

        # 2. 创建特定于此场景的输出目录
        scenario_output_dir = os.path.join(base_config.output_dir, self.scenario_name)
        
        # 3. 初始化所有模型，并将正确的对象传递下去
        #    首先，创建数据加载器实例
        data_loader = TrafficDataLoader(base_config)
        
        #    然后，将 data_loader 对象传递给 TrafficModel
        self.traffic_model = TrafficModel(base_config, data_loader) 
        
        #    最后，将场景配置应用到交通模型中（这一步很重要）
        self.traffic_model.apply_scenario(scenario_config)

        self.power_grid_model = IEEE13BusModel(base_config)
        self.efficiency_model = DynamicBDWPTEfficiency(base_config)
        self.results_analyzer = ResultsAnalyzer(scenario_output_dir, base_config)
        
        # --- ▲▲▲ 修改结束 ▲▲▲ ---
        
        logging.info(f"Simulation Engine initialized for scenario: {self.scenario_name}")

    def run_simulation(self):
        """Runs the main simulation loop for the configured scenario."""
        logging.info(f"Starting simulation from {self.start_time} to {self.end_time}")
        self.results_analyzer.setup_results_file()
        
        current_time = self.start_time
        
        while current_time <= self.end_time:
            logging.debug(f"--- Processing timestep: {current_time} ---")
            
            # 1. 更新交通模型
            day_type = self.scenario_config.get('day_type', 'weekday')
            current_time_minutes = current_time.hour * 60 + current_time.minute
            self.traffic_model.update_vehicle_positions(current_time_minutes, day_type)

            active_vehicles_data = {v['id']: v for v in self.traffic_model.vehicles}

            # 2. 计算BDWPT需求
            bdwpt_demands = self._calculate_bdwpt_demands(active_vehicles_data, current_time)

            # 3. 更新电网负荷
            self.power_grid_model.update_bdwpt_loads(bdwpt_demands['vehicle_powers_kw'])
            
            # 4. 运行潮流计算
            self.power_grid_model.solve_power_flow()

            # 5. [最终确认] 调用我们刚刚添加的 get_grid_state 函数
            grid_state = self.power_grid_model.get_grid_state()
            
            traffic_state = self.traffic_model.get_summary_state()
            
            timestep_data = self._collect_timestep_data(
                current_time, grid_state, traffic_state, bdwpt_demands
            )
            
            self.results_analyzer.log_timeseries_data(timestep_data)
            
            current_time += self.time_step
            
        logging.info(f"Simulation for scenario '{self.scenario_name}' finished.")
        summary = self.results_analyzer.get_summary_statistics()
        self.results_analyzer.finalize(summary)
        
        return {
            'timeseries_file': self.results_analyzer.timeseries_file,
            'summary_file': self.results_analyzer.summary_file
        }

    # _calculate_bdwpt_demands 和 _collect_timestep_data 等辅助方法保持不变...
    def _calculate_bdwpt_demands(self, vehicles, current_time):
        """
        Calculates power demand for each BDWPT vehicle, applying the efficiency model.
        """
        vehicle_powers = {}
        efficiency_details = []

        for v_id, v_data in vehicles.items():
            if v_data.get('is_on_bdwpt_road'):
                power_demand_kw = v_data['power_demand_kw'] # Ideal power
                
                characteristics = BDWPTCharacteristics(
                    coil_alignment_quality=v_data.get('alignment_quality', 0.95),
                    air_gap_mm=v_data.get('air_gap_mm', 100),
                    thermal_state=v_data.get('thermal_state', ThermalState.NORMAL)
                )

                efficiency, details = self.efficiency_model.calculate_efficiency(
                    power_kw=power_demand_kw,
                    characteristics=characteristics,
                    ambient_temp_c=20 
                )
                
                if power_demand_kw > 0 and efficiency > 0:
                    actual_grid_power_kw = power_demand_kw / efficiency
                elif power_demand_kw < 0:
                    actual_grid_power_kw = power_demand_kw * efficiency
                else:
                    actual_grid_power_kw = 0
                
                vehicle_powers[v_id] = actual_grid_power_kw
                efficiency_details.append(details)

        return self._summarize_bdwpt_data(vehicle_powers, efficiency_details)

    def _summarize_bdwpt_data(self, vehicle_powers, efficiency_details):
        """Helper to aggregate BDWPT data for the current timestep."""
        charging_kw = sum(p for p in vehicle_powers.values() if p > 0)
        discharging_kw = sum(p for p in vehicle_powers.values() if p < 0)
        
        num_charging = sum(1 for p in vehicle_powers.values() if p > 0)
        num_discharging = sum(1 for p in vehicle_powers.values() if p < 0)
        
        if not efficiency_details:
            avg_details = {
                'avg_efficiency': 0, 'min_efficiency': 0, 'max_efficiency': 0,
                'power_factor': 0, 'alignment_factor': 0, 'airgap_factor': 0,
                'thermal_factor': 0, 'coupling_factor': 0
            }
        else:
            df = pd.DataFrame(efficiency_details)
            avg_details = {
                'avg_efficiency': df['final_efficiency'].mean(),
                'min_efficiency': df['final_efficiency'].min(),
                'max_efficiency': df['final_efficiency'].max(),
                'power_factor': df['power_factor'].mean(),
                'alignment_factor': df['alignment_factor'].mean(),
                'airgap_factor': df['airgap_factor'].mean(),
                'thermal_factor': df['thermal_factor'].mean(),
                'coupling_factor': df['coupling_factor'].mean()
            }
            
        return {
            'vehicle_powers_kw': vehicle_powers,
            'bdwpt_charging_kw': charging_kw,
            'bdwpt_discharging_kw': abs(discharging_kw),
            'total_bdwpt_kw': charging_kw + discharging_kw,
            'charging_vehicles': num_charging,
            'discharging_vehicles': num_discharging,
            'efficiency_summary': avg_details
        }
        
    def _collect_timestep_data(self, timestamp, grid_state, traffic_state, bdwpt_state):
        """Assembles the complete data dictionary for a single timestep."""
        
        data = {key: 0 for key in ALL_TIMESERIES_COLUMNS}
        
        data['timestamp'] = timestamp.isoformat()
        
        data['total_load_kw'] = grid_state.get('total_load_kw', 0)
        data['total_generation_kw'] = grid_state.get('total_generation_kw', 0)
        losses = grid_state.get('losses', (0,0))
        data['total_losses_kw'] = losses[0] if isinstance(losses, (list, tuple)) else 0
        feeder_power = grid_state.get('feeder_power_kw', (0,0))
        data['feeder_power_p_kw'] = feeder_power[0] if isinstance(feeder_power, (list, tuple)) else 0
        data['feeder_power_q_kvar'] = feeder_power[1] if isinstance(feeder_power, (list, tuple)) else 0
        
        data['active_vehicles'] = traffic_state.get('active_vehicles', 0)
        
        data.update(bdwpt_state['efficiency_summary'])
        data['charging_vehicles'] = bdwpt_state['charging_vehicles']
        data['discharging_vehicles'] = bdwpt_state['discharging_vehicles']
        data['total_bdwpt_kw'] = bdwpt_state['total_bdwpt_kw']
        data['bdwpt_charging_kw'] = bdwpt_state['bdwpt_charging_kw']
        data['bdwpt_discharging_kw'] = bdwpt_state['bdwpt_discharging_kw']
        
        num_charging = bdwpt_state['charging_vehicles']
        data['avg_power_per_charging_vehicle_kw'] = bdwpt_state['bdwpt_charging_kw'] / num_charging if num_charging > 0 else 0
        num_discharging = bdwpt_state['discharging_vehicles']
        data['avg_power_per_discharging_vehicle_kw'] = bdwpt_state['bdwpt_discharging_kw'] / num_discharging if num_discharging > 0 else 0

        data['reverse_power_flow_events'] = grid_state.get('reverse_power_flow_events', 0)
        data['voltage_violations'] = grid_state.get('voltage_violations', 0)
        
        voltages_pu = grid_state.get('voltages_pu', {})
        data['v_bus_632_pu'] = voltages_pu.get('632', 0)
        data['v_bus_633_pu'] = voltages_pu.get('633', 0)
        data['v_bus_634_pu'] = voltages_pu.get('634', 0)
        data['v_bus_671_pu'] = voltages_pu.get('671', 0)
        data['v_bus_675_pu'] = voltages_pu.get('675', 0)
        data['v_bus_680_pu'] = voltages_pu.get('680', 0)
        data['v_bus_692_pu'] = voltages_pu.get('692', 0)
        data['v_bus_650_pu'] = voltages_pu.get('sourcebus', 0)
        
        return data