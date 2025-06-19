# filepath: d:\1st_year_PhD\EEA_2025\EEA2025_v1.4.1\cosimulation\results_analyzer.py
# /3_cosimulation/results_analyzer.py

import pandas as pd
import logging

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    """
    Analyzes and compares simulation results to calculate Key Performance Indicators (KPIs).
    """

    def __init__(self, config):
        self.config = config

    def calculate_kpis(self, all_results):
        """
        Calculate Key Performance Indicators (KPIs) for all scenarios by comparing
        them against their respective baseline (0% penetration).

        Args:
            all_results (dict): A dictionary where keys are scenario names and
                                values are the result dictionaries from the simulation engine.

        Returns:
            dict: A dictionary of calculated KPIs for each scenario.
        """
        kpis = {}

        # First, identify the baseline results for each scenario type
        # Extract base scenario type from scenario name (e.g., "Weekday Peak_0%" -> "Weekday Peak")
        baseline_results = {}
        for scenario_name, results in all_results.items():
            if results['summary']['bdwpt_penetration'] == 0:
                # Extract base scenario name (e.g., "Weekday Peak_0%" -> "Weekday Peak")
                base_scenario = scenario_name.replace("_0%", "")
                baseline_results[base_scenario] = results
                logger.debug(f"Found baseline for '{base_scenario}': {scenario_name}")
        
        if not baseline_results:
            logger.warning("No baseline (0% penetration) scenarios found. KPI calculation will be limited.")
            return kpis

        logger.info(f"Found baselines for: {list(baseline_results.keys())}")

        # Now, calculate KPIs for each scenario relative to its baseline
        for scenario_name, results in all_results.items():
            if results['summary']['bdwpt_penetration'] == 0:
                # For baseline, KPI is just its own value
                kpi = {
                    'scenario': scenario_name,
                    'peak_reduction_kw': 0,
                    'peak_reduction_pct': 0,
                    'voltage_violations': results['summary']['voltage_violations'],
                    'voltage_improvement': 0,
                    'losses_kwh': results['summary']['total_losses_kwh'],
                    'loss_reduction_kwh': 0,
                    'reverse_flow_events': results['summary']['reverse_flow_events'],
                    'energy_from_v2g_kwh': results['summary']['bdwpt_energy_discharged_kwh'],
                    'energy_to_g2v_kwh': results['summary']['bdwpt_energy_charged_kwh'],
                }
                kpis[scenario_name] = kpi
                logger.debug(f"Added baseline KPI for {scenario_name}")
                continue

            # Extract base scenario name for non-baseline scenarios
            # e.g., "Weekday Peak_15%" -> "Weekday Peak"
            base_scenario = "_".join(scenario_name.split("_")[:-1])
            baseline = baseline_results.get(base_scenario)

            if not baseline:
                logger.warning(f"Could not find baseline for scenario: {scenario_name} (looking for baseline: {base_scenario}). Skipping KPI calculation.")
                logger.debug(f"Available baselines: {list(baseline_results.keys())}")
                continue

            # Calculate KPIs
            peak_reduction = baseline['summary']['peak_load'] - results['summary']['peak_load']
            peak_reduction_pct = (peak_reduction / baseline['summary']['peak_load']) * 100 if baseline['summary']['peak_load'] > 0 else 0
            
            voltage_improvement = baseline['summary']['voltage_violations'] - results['summary']['voltage_violations']
            
            loss_reduction = baseline['summary']['total_losses_kwh'] - results['summary']['total_losses_kwh']

            kpi = {
                'scenario': scenario_name,
                'peak_reduction_kw': peak_reduction,
                'peak_reduction_pct': peak_reduction_pct,
                'voltage_violations': results['summary']['voltage_violations'],
                'voltage_improvement': voltage_improvement,
                'losses_kwh': results['summary']['total_losses_kwh'],
                'loss_reduction_kwh': loss_reduction,
                'reverse_flow_events': results['summary']['reverse_flow_events'],
                'energy_from_v2g_kwh': results['summary']['bdwpt_energy_discharged_kwh'],
                'energy_to_g2v_kwh': results['summary']['bdwpt_energy_charged_kwh'],
            }
            kpis[scenario_name] = kpi
            logger.info(f"Calculated KPIs for {scenario_name} relative to baseline {base_scenario}")
        
        logger.info("KPI calculation complete.")
        return kpis