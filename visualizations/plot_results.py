# visualizations/plot_results.py - Visualization functions for results

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging # FIX: Import the logging library

logger = logging.getLogger(__name__) # FIX: Get the logger instance

class Visualizer:
    """Create visualizations for BDWPT simulation results"""
    
    def __init__(self, config):
        self.config = config
        self.output_dir = config.figures_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def plot_load_curves(self, all_results):
        """Plot 24-hour load curves comparison"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
        
        scenarios = ['Weekday Peak', 'Weekend Peak']
        
        for idx, scenario_base in enumerate(scenarios):
            ax = axes[idx]
            
            for penetration in [0, 15, 40]:
                key = f"{scenario_base}_{penetration}%"
                if key in all_results:
                    df = all_results[key]['timeseries']
                    df_resampled = df.set_index('timestamp').resample('15min').mean()
                    label = f"{penetration}% BDWPT" if penetration > 0 else "Baseline"
                    ax.plot(df_resampled.index, df_resampled['total_load_kw'], label=label, linewidth=2)
                    
            ax.set_title(f'{scenario_base} - Total Load Comparison', fontsize=14)
            ax.set_ylabel('Total Load (kW)', fontsize=12)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
        axes[-1].set_xlabel('Time of Day', fontsize=12)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=3))
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'load_curves_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_voltage_profiles(self, all_results):
        """Plot voltage profiles at critical buses"""
        critical_buses = [671, 675, 652, 611]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
        axes = axes.flatten()
        
        for idx, bus in enumerate(critical_buses):
            ax = axes[idx]
            
            for key in ['Weekday Peak_0%', 'Weekday Peak_15%', 'Weekday Peak_40%']:
                if key in all_results:
                    df = all_results[key]['timeseries']
                    voltage_col = f'voltage_bus_{bus}'
                    
                    if voltage_col in df.columns:
                        df_resampled = df.set_index('timestamp').resample('15min').mean()
                        penetration = key.split('_')[1]
                        label = f"{penetration} BDWPT" if penetration != "0%" else "Baseline"
                        ax.plot(df_resampled.index, df_resampled[voltage_col], label=label, linewidth=2)
                        
            ax.axhline(y=1.05, color='r', linestyle='--', alpha=0.5, label='Upper Limit')
            ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.5, label='Lower Limit')
            
            ax.set_title(f'Bus {bus} Voltage Profile', fontsize=12)
            ax.set_ylabel('Voltage (p.u.)', fontsize=10)
            ax.set_ylim(0.94, 1.06)
            ax.legend(loc='best', fontsize=8)
            ax.grid(True, alpha=0.3)
            
        for ax in axes:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
            ax.set_xlabel('Time of Day', fontsize=10)
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'voltage_profiles.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_kpi_comparison(self, kpis):
        """Plot KPI comparison bar charts"""
        kpi_df = pd.DataFrame.from_dict(kpis, orient='index')
        if kpi_df.empty:
            logger.warning("KPI DataFrame is empty, skipping KPI plot.")
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        axes = axes.flatten()

        plot_defs = [
            {'key': 'peak_reduction_kw', 'title': 'Peak Load Reduction', 'ylabel': 'Reduction (kW)', 'color': 'steelblue'},
            {'key': 'voltage_improvement', 'title': 'Voltage Violation Improvement', 'ylabel': 'Violations Reduced', 'color': 'darkgreen'},
            {'key': 'loss_reduction_kwh', 'title': 'Energy Loss Reduction', 'ylabel': 'Reduction (kWh)', 'color': 'darkorange'},
            {'key': 'energy_from_v2g_kwh', 'title': 'V2G Energy Contribution', 'ylabel': 'Energy (kWh)', 'color': 'darkred'}
        ]

        # Filter out baseline scenarios for plotting meaningful comparisons
        plot_df = kpi_df[kpi_df.index.str.contains("0%") == False].copy()
        if plot_df.empty:
            logger.warning("No non-baseline scenarios found for KPI plotting.")
            plt.close()
            return
            
        plot_df['scenario_label'] = plot_df.index.str.replace('_', '\n')
        
        for i, pdef in enumerate(plot_defs):
            ax = axes[i]
            bars = ax.bar(plot_df['scenario_label'], plot_df[pdef['key']], color=pdef['color'])
            ax.set_title(pdef['title'], fontsize=14)
            ax.set_ylabel(pdef['ylabel'], fontsize=12)
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.grid(True, axis='y', linestyle='--', alpha=0.6)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height, f'{height:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout(pad=3.0)
        plt.savefig(os.path.join(self.output_dir, 'kpi_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_bdwpt_heatmap(self, all_results):
        """Plot heatmap of BDWPT power exchange by node and time"""
        key = 'Weekday Peak_40%'
        if key not in all_results: return
            
        df = all_results[key]['timeseries']
        bdwpt_cols = [col for col in df.columns if 'bdwpt_node' in col]
        if not bdwpt_cols: return
            
        nodes = sorted([int(col.split('_')[2]) for col in bdwpt_cols])
        df_hourly = df.set_index('timestamp').resample('h').mean()
        
        power_matrix = pd.DataFrame(index=df_hourly.index.hour, columns=nodes)
        for node in nodes:
            col = f'bdwpt_node_{node}_kw'
            if col in df_hourly.columns:
                power_matrix[node] = df_hourly[col].values
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(power_matrix.T, cmap='RdBu_r', center=0, cbar_kws={'label': 'Power (kW)\n V2G <— 0 —> G2V'})
        
        plt.title('BDWPT Power Exchange Heatmap (40% Penetration, Weekday)', fontsize=14)
        plt.xlabel('Hour of Day', fontsize=12)
        plt.ylabel('Node Number', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'bdwpt_power_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_agent_soc_profiles(self, all_results, num_agents=5):
        """Plot SoC profiles for a sample of agents"""
        logger.warning("Plotting agent SOC profiles requires detailed agent history, which is not yet fully implemented.")

    def plot_agent_power_exchange(self, all_results, num_agents=5):
        """Plot power exchange for a sample of agents"""
        logger.warning("Plotting agent power exchange requires detailed agent history, which is not yet fully implemented.")