# main.py - 修复语法错误版本

import os
import sys
import time
import logging
import numpy as np
import pandas as pd
from datetime import datetime

# 设置日志记录器
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulation.log', mode='w'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 核心导入
from config import SimulationConfig
from traffic_model.data_loader import TrafficDataLoader
from traffic_model.main_traffic import TrafficModel
from power_grid_model.ieee_13_bus_model import IEEE13BusSystem
from cosimulation.simulation_engine import CoSimulationEngine
from cosimulation.scenarios import ScenarioManager
from cosimulation.results_analyzer import ResultsAnalyzer
from visualizations.plot_results import Visualizer

# 全局变量声明
ENHANCED_FEATURES_AVAILABLE = False

# 尝试导入增强功能
try:
    from visualizations.enhanced_visualizations import EnhancedVisualizer
    from validation.model_validator import ModelValidator
    ENHANCED_FEATURES_AVAILABLE = True
    logger.info("Enhanced features loaded successfully")
except ImportError as e:
    logger.warning(f"Enhanced features not available: {e}")
    logger.info("Continuing with basic functionality...")

class BDWPTSimulationPlatform:
    """Main simulation platform for BDWPT in urban LV networks"""
    
    def __init__(self):
        self.config = SimulationConfig()
        self.traffic_model = None
        self.power_grid = None
        self.cosim_engine = None
        self.scenario_manager = None
        self.results_analyzer = None
        self.visualizer = None
        
        # 增强功能组件
        if ENHANCED_FEATURES_AVAILABLE:
            self.enhanced_visualizer = None
            self.model_validator = None
        
    def initialize(self):
        """Initialize all simulation components"""
        logger.info("Initializing BDWPT Simulation Platform...")
        
        try:
            # 确保输出目录存在
            os.makedirs(self.config.output_dir, exist_ok=True)
            os.makedirs(self.config.figures_dir, exist_ok=True)
            os.makedirs(self.config.results_dir, exist_ok=True)
            
            logger.info("Setting up data loader...")
            data_loader = TrafficDataLoader(self.config.data_dir)

            logger.info("Setting up traffic model...")
            self.traffic_model = TrafficModel(self.config, data_loader)
            
            logger.info("Setting up IEEE 13-bus test system...")
            self.power_grid = IEEE13BusSystem(self.config)
            self.power_grid.build_network()
            
            logger.info("Setting up co-simulation engine...")
            self.cosim_engine = CoSimulationEngine(
                self.config,
                self.traffic_model,
                self.power_grid
            )
            
            self.scenario_manager = ScenarioManager(self.config)
            self.results_analyzer = ResultsAnalyzer(self.config)
            self.visualizer = Visualizer(self.config)
            
            # 初始化增强功能
            if ENHANCED_FEATURES_AVAILABLE:
                logger.info("Initializing enhanced features...")
                try:
                    self.enhanced_visualizer = EnhancedVisualizer(
                        self.config, self.config.figures_dir
                    )
                    self.model_validator = ModelValidator(self.config)
                    logger.info("Enhanced features initialized successfully")
                except Exception as e:
                    logger.warning(f"Failed to initialize enhanced features: {e}")
                    # 注意：这里不再修改全局变量，而是设置实例变量
                    self.enhanced_visualizer = None
                    self.model_validator = None
            
            logger.info("Initialization complete!")
            
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def run_scenario(self, scenario_name, bdwpt_penetration):
        """Run a single scenario with specified BDWPT penetration"""
        logger.info(f"Running scenario: {scenario_name} with {bdwpt_penetration}% BDWPT penetration")
        
        try:
            scenario = self.scenario_manager.get_scenario(scenario_name, bdwpt_penetration)
            
            start_time = time.time()
            results = self.cosim_engine.run_simulation(scenario)
            elapsed_time = time.time() - start_time
            
            logger.info(f"Simulation completed in {elapsed_time:.2f} seconds")
            return results
            
        except Exception as e:
            logger.error(f"Error in scenario {scenario_name}_{bdwpt_penetration}%: {e}")
            raise

    def run_all_scenarios(self):
        """Run all scenarios"""
        all_results = {}
        scenarios_to_run = self.scenario_manager.get_all_scenarios_to_run()
        
        logger.info(f"Starting simulation of {len(scenarios_to_run)} scenarios...")
        
        for i, scenario in enumerate(scenarios_to_run, 1):
            key = scenario['name']
            logger.info(f"\n{'='*60}")
            logger.info(f"Starting scenario {i}/{len(scenarios_to_run)}: {key}")
            logger.info(f"{'='*60}")
            
            try:
                results = self.run_scenario(scenario['base_name'], scenario['bdwpt_penetration'])
                all_results[key] = results
                self.save_results(results, key)
                logger.info(f"✅ Scenario {key} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ FATAL ERROR in scenario {key}: {str(e)}")
                import traceback
                logger.error(f"Full traceback: {traceback.format_exc()}")
                # 继续执行其他场景而不是完全停止
                continue
                
        return all_results

    def analyze_results(self, all_results):
        """Analyze simulation results and calculate KPIs"""
        logger.info("\nAnalyzing simulation results...")
        
        if not all_results:
            logger.warning("No results to analyze!")
            return {}
        
        try:
            kpis = self.results_analyzer.calculate_kpis(all_results)
            
            logger.info("\n" + "="*25 + " KEY PERFORMANCE INDICATORS " + "="*25)
            kpi_data = []
            for scenario, metrics in kpis.items():
                kpi_data.append({
                    'Scenario': scenario,
                    'Peak Reduction (kW)': metrics.get('peak_reduction_kw', 0),
                    'Loss Reduction (kWh)': metrics.get('loss_reduction_kwh', 0),
                    'V2G Energy (kWh)': metrics.get('energy_from_v2g_kwh', 0),
                    'Voltage Improvement': metrics.get('voltage_improvement', 0)
                })
            
            if kpi_data:
                kpi_df = pd.DataFrame(kpi_data)
                logger.info("\n" + kpi_df.to_string())
            
            logger.info("="*78)
            return kpis
            
        except Exception as e:
            logger.error(f"Error analyzing results: {e}")
            return {}

    def generate_visualizations(self, all_results, kpis):
        """Generate all required visualizations"""
        logger.info("\nGenerating visualizations...")
        
        try:
            figures_path = os.path.abspath(self.config.figures_dir)
            os.makedirs(figures_path, exist_ok=True)
            logger.info(f"Figures directory: {figures_path}")

            # 基础可视化
            self.visualizer.plot_load_curves(all_results)
            self.visualizer.plot_voltage_profiles(all_results)
            self.visualizer.plot_kpi_comparison(kpis)
            self.visualizer.plot_bdwpt_heatmap(all_results)
            
            # 增强可视化
            if ENHANCED_FEATURES_AVAILABLE and self.enhanced_visualizer:
                logger.info("Generating enhanced visualizations...")
                try:
                    enhanced_files = self.enhanced_visualizer.generate_all_visualizations(
                        all_results, kpis
                    )
                    logger.info(f"Generated {len(enhanced_files)} enhanced visualization files")
                except Exception as e:
                    logger.warning(f"Enhanced visualization generation failed: {e}")
            
            logger.info(f"✅ Visualizations saved to: {figures_path}")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")

    def save_results(self, results, scenario_name):
        """Save simulation results to files"""
        try:
            scenario_dir = os.path.join(
                self.config.results_dir, 
                scenario_name.replace(' ', '_').replace('%', 'pct')
            )
            os.makedirs(scenario_dir, exist_ok=True)
            
            if 'timeseries' in results and isinstance(results['timeseries'], pd.DataFrame):
                csv_path = os.path.join(scenario_dir, 'timeseries_data.csv')
                results['timeseries'].to_csv(csv_path, index=False)
                
            if 'summary' in results:
                summary_path = os.path.join(scenario_dir, 'summary_statistics.txt')
                with open(summary_path, 'w') as f:
                    for key, value in results['summary'].items():
                        f.write(f"{key}: {value}\n")
            
            logger.debug(f"Results saved to: {scenario_dir}")
            
        except Exception as e:
            logger.error(f"Error saving results for {scenario_name}: {e}")

    def run_model_validation(self):
        """Run comprehensive model validation"""
        if not ENHANCED_FEATURES_AVAILABLE or not self.model_validator:
            logger.info("Model validation not available")
            return None
            
        logger.info("\n" + "="*25 + " MODEL VALIDATION " + "="*25)
        
        try:
            validation_report = self.model_validator.run_comprehensive_validation()
            
            summary = self.model_validator.get_validation_summary()
            logger.info(summary)
            
            report_path = os.path.join(self.config.output_dir, "model_validation_report.json")
            self.model_validator.export_validation_report(report_path)
            logger.info(f"Detailed validation report saved to: {report_path}")
            
            return validation_report
            
        except Exception as e:
            logger.error(f"Model validation failed: {e}")
            return None

    def run(self):
        """Main execution method"""
        try:
            logger.info("🚀 Starting BDWPT Simulation Platform")
            
            # 可选的模型验证
            if ENHANCED_FEATURES_AVAILABLE:
                logger.info("Running pre-simulation model validation...")
                validation_report = self.run_model_validation()
                
                if validation_report and validation_report.get('overall_score', 0) < 0.7:
                    logger.warning("⚠️  Model validation score is low. Consider reviewing parameters.")
                logger.info("Continuing with simulation...")
            
            # 主要仿真流程
            self.initialize()
            all_results = self.run_all_scenarios()
            
            if all_results:
                kpis = self.analyze_results(all_results)
                self.generate_visualizations(all_results, kpis)
                
                logger.info("\n" + "="*80)
                logger.info("🎉 SIMULATION COMPLETED SUCCESSFULLY!")
                logger.info(f"📁 Results available in: {self.config.output_dir}")
                logger.info(f"📊 Visualizations in: {self.config.figures_dir}")
                logger.info("="*80)
            else:
                logger.warning("❌ No scenarios completed successfully!")
            
        except Exception as e:
            logger.error(f"💥 Simulation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

def main():
    """Main entry point"""
    try:
        platform = BDWPTSimulationPlatform()
        platform.run()
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()