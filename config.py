# config.py
# Configuration file for BDWPT simulation platform

import os
import numpy as np
import logging
from datetime import datetime, timedelta

# 获取此配置文件所在的目录的绝对路径
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

logger = logging.getLogger(__name__)

class SimulationConfig:
    """Configuration parameters for the BDWPT simulation."""
    
    def __init__(self):
        # 构建所有路径从绝对基础路径
        self.data_dir = os.path.join(_BASE_DIR, "data")
        self.output_dir = os.path.join(_BASE_DIR, "output")
        self.figures_dir = os.path.join(self.output_dir, "figures")
        self.results_dir = os.path.join(self.output_dir, "results")
        self.logs_dir = os.path.join(self.output_dir, "logs")
        
        # 确保目录存在
        for directory in [self.data_dir, self.output_dir, self.figures_dir, self.results_dir, self.logs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Simulation parameters
        self.simulation_params = {
            'start_time': datetime(2024, 1, 1, 0, 0),  # Start at midnight
            'end_time': datetime(2024, 1, 1, 23, 59),  # End at 23:59
            'time_step_minutes': 15,  # 15-minute time steps
            'day_types': ['weekday', 'weekend']
        }
        
        # Traffic model parameters
        self.traffic_params = {
            'total_vehicles': 1000,
            'ev_penetration': 0.3,  # 30% EV penetration
            'trips_per_vehicle_per_day': 2.5,  # Average trips per vehicle per day
            'trip_generation_rate': 2.5,  # Average trips per vehicle per day
            'peak_hour_factor': 1.5,
            'speed_limit_kmh': 50,
            'average_trip_distance_km': 8.5  # Average trip distance in km
        }
        
        # Add time step convenience property
        self.time_step_minutes = self.simulation_params['time_step_minutes']
        
        # EV parameters
        self.ev_params = {
            'initial_soc_mean': 0.7,  # 70% average initial SOC
            'initial_soc_std': 0.15,  # 15% standard deviation
            'charging_efficiency': 0.9,  # 90% charging efficiency
            'energy_consumption_kwh_per_km': 0.15,  # 150 Wh/km (alias for compatibility)
            'min_soc_threshold': 0.2,  # 20% minimum SOC
            'max_soc_threshold': 0.9   # 90% maximum SOC for normal charging
        }
        
        # BDWPT system parameters
        self.bdwpt_params = {
            'max_power_kw': 50,  # Maximum BDWPT power per vehicle
            'charging_power_kw': 50,  # Charging power
            'discharging_power_kw': 30,  # Discharging power for V2G
            'efficiency': 0.85,  # 85% wireless power transfer efficiency
            'activation_distance_m': 5,  # Distance for BDWPT activation
            'min_vehicle_speed_kmh': 5,  # Minimum speed for BDWPT operation
            'max_vehicle_speed_kmh': 60,  # Maximum speed for BDWPT operation
            'power_control_algorithm': 'voltage_regulation'
        }
        
        # BDWPT control parameters
        self.control_params = {
            'soc_force_charge': 0.2,  # Force charging below this SoC
            'soc_force_discharge': 0.9,  # Force discharging above this SoC
            'soc_min_v2g': 0.3,  # Minimum SoC for V2G operation
            'voltage_critical_high': 1.05,  # Critical high voltage (p.u.)
            'voltage_critical_low': 0.95,  # Critical low voltage (p.u.)
            'voltage_high_threshold': 1.02,  # High voltage threshold
            'voltage_low_threshold': 0.98,  # Low voltage threshold
            'tariff_high_threshold': 20.0,  # High tariff threshold (cents/kWh)
            'tariff_low_threshold': 15.0,  # Low tariff threshold (cents/kWh)
            'hysteresis_factor': 0.1  # Hysteresis factor for mode switching
        }
        
        # Power grid parameters (IEEE 13-bus system)
        self.grid_params = {
            'base_voltage_kv': 4.16,  # 4.16 kV base voltage
            'base_power_mva': 5.0,    # 5 MVA base power
            'voltage_tolerance': 0.05,  # ±5% voltage tolerance
            'max_loading_percent': 80,  # 80% maximum loading
            'bdwpt_nodes': [632, 633, 634, 645, 646, 671, 675, 680],  # IEEE 13-bus node numbers
            'bdwpt_connection_type': 'three_phase'
        }
        
        # Scenario configuration
        self.scenario_params = {
            'base_case': {
                'bdwpt_penetration': 0,
                'description': 'Baseline scenario without BDWPT'
            },
            'low_penetration': {
                'bdwpt_penetration': 10,
                'description': '10% BDWPT penetration'
            },
            'medium_penetration': {
                'bdwpt_penetration': 25,
                'description': '25% BDWPT penetration'
            },
            'high_penetration': {
                'bdwpt_penetration': 50,
                'description': '50% BDWPT penetration'
            }
        }
        
        # Penetration scenarios list for easy iteration
        self.penetration_scenarios = [0, 15, 40]
        
        # Base scenarios configuration
        self.scenarios = {
            'Weekday Peak': {
                'load_profile': 'weekday_peak',
                'traffic_multiplier': 1.5,
                'description': 'Weekday peak hours scenario'
            },
            'Weekday Off-Peak': {
                'load_profile': 'weekday_offpeak',
                'traffic_multiplier': 0.8,
                'description': 'Weekday off-peak hours scenario'
            },
            'Weekend Peak': {
                'load_profile': 'weekend_peak',
                'traffic_multiplier': 1.2,
                'description': 'Weekend peak hours scenario'
            },
            'Weekend': {
                'load_profile': 'weekend',
                'traffic_multiplier': 1.0,
                'description': 'Weekend scenario'
            }
        }
        
        # Data file paths
        self.data_paths = {
            'ev_registrations': os.path.join(self.data_dir, 'ev_registrations.csv'),
            'road_network': os.path.join(self.data_dir, 'wellington_roads.json'),
            'load_profiles': os.path.join(self.data_dir, 'load_profiles.csv'),
            'weather_data': os.path.join(self.data_dir, 'weather_data.csv')
        }
        
        # Output paths
        self.output_paths = {
            'results': self.results_dir,
            'figures': self.figures_dir,
            'logs': self.logs_dir
        }
        
        # Logging configuration
        self.logging_config = {
            'level': 'DEBUG',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'simulation.log'
        }

    def get_time_steps(self):
        """Generate list of time steps for the simulation."""
        time_steps = []
        current_time = self.simulation_params['start_time']
        end_time = self.simulation_params['end_time']
        step_delta = timedelta(minutes=self.simulation_params['time_step_minutes'])
        
        while current_time <= end_time:
            time_steps.append(current_time)
            current_time += step_delta
        
        return time_steps

    def get_time_step_minutes(self, time_step):
        """Convert datetime to minutes from start of day."""
        start_of_day = time_step.replace(hour=0, minute=0, second=0, microsecond=0)
        return int((time_step - start_of_day).total_seconds() / 60)

    def get_day_type(self, date):
        """Determine if the given date is a weekday or weekend."""
        return 'weekend' if date.weekday() >= 5 else 'weekday'

    def get_load_profile(self, node_id, time_minutes, day_type):
        """Get load profile for a specific node and time."""
        hour = time_minutes // 60
        
        if day_type == 'weekend':
            base_factor = 0.6 + 0.3 * np.sin(2 * np.pi * (hour - 8) / 24)
        else:
            morning_peak = 0.8 * np.exp(-((hour - 8) ** 2) / 8)
            evening_peak = 1.0 * np.exp(-((hour - 18) ** 2) / 12)
            base_factor = 0.4 + morning_peak + evening_peak
        
        node_factors = {
            632: 1.2, 633: 0.8, 634: 1.0, 645: 0.9, 646: 1.1,
            671: 0.7, 675: 1.3, 680: 0.6
        }
        node_factor = node_factors.get(node_id, 1.0)
        base_load_kw = 100
        
        return base_load_kw * base_factor * node_factor

    def get_time_series(self):
        """Generate time series for simulation based on configuration."""
        time_steps = self.get_time_steps()
        return {
            'time_steps': time_steps,
            'time_minutes': [self.get_time_step_minutes(ts) for ts in time_steps],
            'total_steps': len(time_steps)
        }

    def get_tariff_at_hour(self, hour):
        """Get electricity tariff rate for a specific hour."""
        if 6 <= hour < 10 or 17 <= hour < 21:
            return 25.0
        elif 10 <= hour < 17:
            return 18.0
        else:
            return 12.0

    def validate_config(self):
        """Validate configuration parameters."""
        errors = []
        warnings = []
        
        # 验证BDWPT节点
        if not all(isinstance(node, int) for node in self.grid_params['bdwpt_nodes']):
            errors.append("BDWPT nodes must be integers")
        
        # 验证节点数量
        if len(self.grid_params['bdwpt_nodes']) == 0:
            errors.append("At least one BDWPT node must be specified")
        
        # 验证渗透率场景
        if not all(0 <= p <= 100 for p in self.penetration_scenarios):
            errors.append("Penetration scenarios must be between 0 and 100")
        
        # 验证时间设置
        if self.simulation_params['start_time'] >= self.simulation_params['end_time']:
            errors.append("Start time must be before end time")
        
        # 验证时间步长
        if self.simulation_params['time_step_minutes'] <= 0:
            errors.append("Time step must be positive")
        elif self.simulation_params['time_step_minutes'] > 60:
            warnings.append("Time step larger than 60 minutes may reduce accuracy")
        
        # 验证车辆参数
        if self.traffic_params['total_vehicles'] <= 0:
            errors.append("Total vehicles must be positive")
        
        if not 0 <= self.traffic_params['ev_penetration'] <= 1:
            errors.append("EV penetration must be between 0 and 1")
        
        # 验证SOC设置
        if not 0 <= self.ev_params['initial_soc_mean'] <= 1:
            errors.append("Initial SOC mean must be between 0 and 1")
        
        if self.ev_params['initial_soc_std'] < 0:
            errors.append("Initial SOC standard deviation must be non-negative")
        
        if self.ev_params['min_soc_threshold'] >= self.ev_params['max_soc_threshold']:
            errors.append("Min SOC threshold must be less than max SOC threshold")
        
        # 验证功率设置
        if self.bdwpt_params['max_power_kw'] <= 0:
            errors.append("BDWPT max power must be positive")
        
        if self.bdwpt_params['charging_power_kw'] <= 0:
            errors.append("BDWPT charging power must be positive")
        
        if self.bdwpt_params['discharging_power_kw'] <= 0:
            errors.append("BDWPT discharging power must be positive")
        
        # 验证效率设置
        if not 0 < self.bdwpt_params['efficiency'] <= 1:
            errors.append("BDWPT efficiency must be between 0 and 1")
        
        if not 0 < self.ev_params['charging_efficiency'] <= 1:
            errors.append("EV charging efficiency must be between 0 and 1")
        
        # 验证电网参数
        if self.grid_params['base_voltage_kv'] <= 0:
            errors.append("Base voltage must be positive")
        
        if not 0 < self.grid_params['voltage_tolerance'] < 1:
            errors.append("Voltage tolerance must be between 0 and 1")
        
        # 验证控制参数
        control_soc_params = ['soc_force_charge', 'soc_force_discharge', 'soc_min_v2g']
        for param in control_soc_params:
            if not 0 <= self.control_params[param] <= 1:
                errors.append(f"Control parameter {param} must be between 0 and 1")
        
        # 验证电压阈值
        voltage_params = ['voltage_critical_high', 'voltage_critical_low', 
                         'voltage_high_threshold', 'voltage_low_threshold']
        for param in voltage_params:
            if self.control_params[param] <= 0:
                errors.append(f"Voltage parameter {param} must be positive")
        
        # 验证目录存在性
        critical_dirs = [self.output_dir, self.figures_dir, self.results_dir]
        for dir_path in critical_dirs:
            if not os.path.exists(dir_path):
                try:
                    os.makedirs(dir_path, exist_ok=True)
                    logger.info(f"Created directory: {dir_path}")
                except Exception as e:
                    errors.append(f"Cannot create directory {dir_path}: {e}")
        
        # 验证场景配置
        if not self.scenarios:
            errors.append("At least one scenario must be defined")
        
        for scenario_name, scenario_config in self.scenarios.items():
            if 'load_profile' not in scenario_config:
                warnings.append(f"Scenario '{scenario_name}' missing load_profile")
            if 'traffic_multiplier' not in scenario_config:
                warnings.append(f"Scenario '{scenario_name}' missing traffic_multiplier")
        
        # 验证能耗参数合理性
        if self.ev_params['energy_consumption_kwh_per_km'] <= 0:
            errors.append("Energy consumption must be positive")
        elif self.ev_params['energy_consumption_kwh_per_km'] > 1.0:
            warnings.append("Energy consumption > 1 kWh/km seems high for modern EVs")
        
        # 验证行程参数
        if self.traffic_params['trips_per_vehicle_per_day'] <= 0:
            errors.append("Trips per vehicle per day must be positive")
        elif self.traffic_params['trips_per_vehicle_per_day'] > 10:
            warnings.append("More than 10 trips per vehicle per day seems unrealistic")
        
        # 报告结果
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(f"❌ {error}" for error in errors)
            raise ValueError(error_msg)
        
        if warnings:
            logger.warning("Configuration warnings:")
            for warning in warnings:
                logger.warning(f"⚠️  {warning}")
        
        logger.info("✅ Configuration validation passed")
        return True

    def get_required_packages(self):
        """Get list of required Python packages"""
        return [
            'numpy>=1.20.0',
            'pandas>=1.3.0',
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'scipy>=1.7.0',
            'tqdm>=4.60.0'
        ]

    def check_dependencies(self):
        """Check if all required packages are available"""
        missing_packages = []
        package_versions = {}
        
        try:
            import numpy
            package_versions['numpy'] = numpy.__version__
        except ImportError:
            missing_packages.append('numpy')
        
        try:
            import pandas
            package_versions['pandas'] = pandas.__version__
        except ImportError:
            missing_packages.append('pandas')
        
        try:
            import matplotlib
            package_versions['matplotlib'] = matplotlib.__version__
        except ImportError:
            missing_packages.append('matplotlib')
        
        try:
            import seaborn
            package_versions['seaborn'] = seaborn.__version__
        except ImportError:
            missing_packages.append('seaborn')
        
        try:
            import scipy
            package_versions['scipy'] = scipy.__version__
        except ImportError:
            missing_packages.append('scipy')
        
        try:
            import tqdm
            package_versions['tqdm'] = tqdm.__version__
        except ImportError:
            missing_packages.append('tqdm')
        
        # 可选依赖
        optional_packages = {}
        try:
            import py_dss_interface
            optional_packages['py_dss_interface'] = "Available"
        except ImportError:
            optional_packages['py_dss_interface'] = "Not installed (using simplified power flow)"
        
        if missing_packages:
            logger.error("❌ Missing required packages:")
            for package in missing_packages:
                logger.error(f"   - {package}")
            logger.info("Install missing packages with:")
            logger.info(f"   pip install {' '.join(missing_packages)}")
            return False
        else:
            logger.info("✅ All required packages available:")
            for package, version in package_versions.items():
                logger.info(f"   - {package}: {version}")
            
            if optional_packages:
                logger.info("📦 Optional packages:")
                for package, status in optional_packages.items():
                    logger.info(f"   - {package}: {status}")
            
            return True

    def get_system_info(self):
        """Get system information for debugging"""
        import platform
        import sys
        
        info = {
            'platform': platform.platform(),
            'python_version': sys.version,
            'python_executable': sys.executable,
            'working_directory': os.getcwd(),
            'config_base_dir': _BASE_DIR,
            'data_dir': self.data_dir,
            'output_dir': self.output_dir,
            'total_scenarios': len(self.scenarios) * len(self.penetration_scenarios)
        }
        
        return info

    def print_config_summary(self):
        """Print a summary of the configuration"""
        logger.info("\n" + "="*60)
        logger.info("🔧 BDWPT SIMULATION CONFIGURATION SUMMARY")
        logger.info("="*60)
        
        logger.info(f"📁 Base Directory: {_BASE_DIR}")
        logger.info(f"📊 Data Directory: {self.data_dir}")
        logger.info(f"📈 Output Directory: {self.output_dir}")
        
        logger.info(f"\n⏰ Simulation Parameters:")
        logger.info(f"   • Start Time: {self.simulation_params['start_time']}")
        logger.info(f"   • End Time: {self.simulation_params['end_time']}")
        logger.info(f"   • Time Step: {self.simulation_params['time_step_minutes']} minutes")
        
        logger.info(f"\n🚗 Traffic Parameters:")
        logger.info(f"   • Total Vehicles: {self.traffic_params['total_vehicles']}")
        logger.info(f"   • EV Penetration: {self.traffic_params['ev_penetration']:.1%}")
        logger.info(f"   • Avg Trips/Day: {self.traffic_params['trips_per_vehicle_per_day']}")
        
        logger.info(f"\n🔋 BDWPT Parameters:")
        logger.info(f"   • Max Power: {self.bdwpt_params['max_power_kw']} kW")
        logger.info(f"   • Efficiency: {self.bdwpt_params['efficiency']:.1%}")
        logger.info(f"   • Nodes: {self.grid_params['bdwpt_nodes']}")
        
        logger.info(f"\n📋 Scenarios to Run:")
        total_scenarios = len(self.scenarios) * len(self.penetration_scenarios)
        logger.info(f"   • Base Scenarios: {list(self.scenarios.keys())}")
        logger.info(f"   • Penetration Levels: {self.penetration_scenarios}")
        logger.info(f"   • Total Scenarios: {total_scenarios}")
        
        logger.info("="*60)

# Create global config instance
config = SimulationConfig()

# Validate configuration on import
if __name__ != "__main__":
    try:
        config.validate_config()
    except ValueError as e:
        logger.error(f"Configuration validation failed: {e}")
        raise