# /3_cosimulation/scenarios.py

import logging

logger = logging.getLogger(__name__)

class ScenarioManager:
    """
    Manages the creation and retrieval of simulation scenarios based on the configuration.
    """
    def __init__(self, config):
        """
        Initializes the ScenarioManager.
        Args:
            config (SimulationConfig): The main configuration object.
        """
        self.config = config
        self.base_scenarios = self.config.scenarios

    def get_scenario(self, base_name, bdwpt_penetration):
        """
        Creates a specific scenario configuration dictionary.

        Args:
            base_name (str): The name of the base scenario (e.g., 'Weekday Peak').
            bdwpt_penetration (int): The penetration level of BDWPT in percent (0, 15, 40).

        Returns:
            dict: A dictionary containing the full configuration for the specific scenario.
        """
        if base_name not in self.base_scenarios:
            raise ValueError(f"Base scenario '{base_name}' not found in config.")

        base_config = self.base_scenarios[base_name]
        scenario_config = {
            'name': f"{base_name}_{bdwpt_penetration}%",
            'base_name': base_name,
            'bdwpt_penetration': bdwpt_penetration,
            'day_type': 'weekday' if 'Weekday' in base_name else 'weekend',
            'load_profile': base_config.get('load_profile', 'weekday'),
            'traffic_multiplier': base_config.get('traffic_multiplier', 1.0),
        }
        
        logger.debug(f"Generated scenario config: {scenario_config['name']}")
        return scenario_config

    def get_all_scenarios_to_run(self):
        """
        Generates a list of all scenario configurations to be simulated.

        Returns:
            list: A list of scenario configuration dictionaries.
        """
        scenarios_to_run = []
        penetration_levels = self.config.penetration_scenarios
        
        for base_name in self.base_scenarios:
            for penetration in penetration_levels:
                scenarios_to_run.append(self.get_scenario(base_name, penetration))
                
        return scenarios_to_run