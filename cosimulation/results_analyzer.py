# EEA2025_v1.4.5_fixed/cosimulation/results_analyzer.py

import os
import csv
import logging
import json
import pandas as pd

# Define all expected columns for the timeseries data. This is the single source of truth.
ALL_TIMESERIES_COLUMNS = [
    'timestamp', 'total_load_kw', 'total_generation_kw', 'total_losses_kw', 
    'feeder_power_p_kw', 'feeder_power_q_kvar',
    'active_vehicles', 'charging_vehicles', 'discharging_vehicles',
    'total_bdwpt_kw', 'bdwpt_charging_kw', 'bdwpt_discharging_kw',
    'avg_efficiency', 'min_efficiency', 'max_efficiency',
    'reverse_power_flow_events', 'voltage_violations',
    # Add specific bus voltages as requested (example buses)
    'v_bus_632_pu', 'v_bus_633_pu', 'v_bus_634_pu', 'v_bus_671_pu',
    'v_bus_675_pu', 'v_bus_680_pu', 'v_bus_692_pu', 'v_bus_650_pu',
    # Add more detailed BDWPT stats
    'avg_power_per_charging_vehicle_kw', 'avg_power_per_discharging_vehicle_kw',
    'efficiency_power_factor', 'efficiency_alignment_factor', 'efficiency_airgap_factor',
    'efficiency_thermal_factor', 'efficiency_coupling_factor'
]

class ResultsAnalyzer:
    """Handles the collection, processing, and output of simulation results."""

    def __init__(self, output_dir, config):
        self.output_dir = output_dir
        self.config = config
        self.timeseries_file = os.path.join(self.output_dir, 'timeseries_data.csv')
        self.summary_file = os.path.join(self.output_dir, 'summary_statistics.txt')
        self.writer = None
        self.file = None
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        logging.info(f"Results will be saved in: {self.output_dir}")

    def setup_results_file(self):
        """Sets up the CSV file for timeseries data using DictWriter."""
        try:
            self.file = open(self.timeseries_file, 'w', newline='', encoding='utf-8')
            # Use DictWriter for robust column handling
            self.writer = csv.DictWriter(self.file, fieldnames=ALL_TIMESERIES_COLUMNS)
            self.writer.writeheader()
            logging.info(f"Timeseries CSV file created with {len(ALL_TIMESERIES_COLUMNS)} columns.")
        except IOError as e:
            logging.error(f"Failed to open results file {self.timeseries_file}: {e}")
            raise

    def log_timeseries_data(self, data_dict):
        """Logs a complete dictionary of data for a single timestep."""
        if self.writer:
            try:
                # DictWriter will handle missing keys, filling them with empty strings.
                # It's better to ensure the input dict is complete.
                self.writer.writerow(data_dict)
            except Exception as e:
                logging.error(f"Error writing timeseries data: {e}")

    def finalize(self, summary_stats):
        """Writes summary statistics and closes the results file."""
        if self.file and not self.file.closed:
            self.file.close()
            logging.info("Timeseries data file closed.")

        try:
            with open(self.summary_file, 'w', encoding='utf-8') as f:
                f.write("--- Simulation Summary Statistics ---\n\n")
                for key, value in summary_stats.items():
                    if isinstance(value, pd.Series):
                        f.write(f"{key}:\n{value.to_string()}\n\n")
                    elif isinstance(value, dict):
                         f.write(f"{key}:\n")
                         for sub_key, sub_value in value.items():
                             f.write(f"  {sub_key}: {sub_value}\n")
                         f.write("\n")
                    else:
                        f.write(f"{key}: {value}\n")
            logging.info(f"Summary statistics saved to {self.summary_file}")
        except IOError as e:
            logging.error(f"Failed to write summary file {self.summary_file}: {e}")
            
    def get_summary_statistics(self):
        """
        Calculates and returns summary statistics from the generated timeseries data.
        This is called after the simulation is complete.
        """
        try:
            df = pd.read_csv(self.timeseries_file)
            summary = {
                "Simulation Duration": f"{(pd.to_datetime(df['timestamp'].iloc[-1]) - pd.to_datetime(df['timestamp'].iloc[0]))}",
                "Total Timesteps": len(df),
                "Grid Loading (kW)": df['total_load_kw'].describe(),
                "Grid Losses (kW)": df['total_losses_kw'].describe(),
                "BDWPT Charging (kW)": df['bdwpt_charging_kw'][df['bdwpt_charging_kw'] > 0].describe(),
                "BDWPT Discharging (kW)": df['bdwpt_discharging_kw'][df['bdwpt_discharging_kw'] > 0].describe(),
                "Overall Efficiency (%)": (df['avg_efficiency'] * 100).describe(),
                "Active Vehicles": df['active_vehicles'].describe(),
                "Final Reverse Power Flow Events": df['reverse_power_flow_events'].iloc[-1],
                "Final Voltage Violations": df['voltage_violations'].iloc[-1]
            }
            return summary
        except (FileNotFoundError, pd.errors.EmptyDataError) as e:
            logging.error(f"Cannot generate summary, timeseries file is missing or empty: {e}")
            return {"Error": "Could not generate summary statistics."}