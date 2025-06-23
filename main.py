# main.py - 完整修复版本

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
        logging.FileHandler('simulation.log', mode='w', encoding='utf-8'), # 添加UTF-8编码
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# --- 核心导入 ---
# [修复] 统一使用修正后的类名
from config import SimulationConfig
from traffic_model.data_loader import TrafficDataLoader
from traffic_model.main_traffic import TrafficModel
from power_grid_model.ieee_13_bus_model import IEEE13BusModel 
from cosimulation.simulation_engine import SimulationEngine
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
    """主仿真平台，负责协调所有模块"""

    def __init__(self):
        """初始化仿真平台"""
        self.config = SimulationConfig()
        self.traffic_data_loader = None
        self.traffic_model = None
        self.power_grid = None
        self.engine = None
        self.scenario_manager = None
        self.results_analyzer = None
        self.visualizer = None

    def initialize(self):
        """初始化所有必要的组件"""
        logger.info("Initializing BDWPT Simulation Platform...")

        logger.info("Setting up data loader...")
        self.traffic_data_loader = TrafficDataLoader(self.config)
        
        logger.info("Setting up traffic model...")
        self.traffic_model = TrafficModel(self.config, self.traffic_data_loader)

        logger.info("Setting up IEEE 13-bus test system...")
        # [修复] 使用与导入语句一致的正确类名 IEEE13BusModel
        self.power_grid = IEEE13BusModel(self.config)
        
        logger.info("Setting up scenario manager...")
        self.scenario_manager = ScenarioManager(self.config)
        
        logger.info("Setting up results analyzer...")
        self.results_analyzer = ResultsAnalyzer(self.config.output_dir, self.config)
        
        logger.info("Setting up visualizer...")
        self.visualizer = Visualizer(self.config)
        
        logger.info("All components initialized successfully.")

    def run_all_scenarios(self):
        """运行所有定义的场景"""
        scenarios = self.scenario_manager.get_all_scenarios_to_run()
        all_results = {}

        # --- ▼▼▼ 这里是修改的核心 ▼▼▼ ---
        # 1. 直接遍历列表 `scenarios`，每一次循环得到一个 `scenario_config` 字典
        for scenario_config in scenarios:
            # 2. 从 `scenario_config` 字典中，通过键 'name' 获取场景的名称
            scenario_name = scenario_config.get('name', 'Unnamed Scenario')
            
            logger.info(f"\n" + "-"*80)
            logger.info(f"[SCENARIO START] Running scenario: {scenario_name}")
            logger.info(f"Scenario config: {scenario_config}")
            
            try:
                # 3. 创建引擎实例时，传入的是完整的场景配置字典 `scenario_config`
                #    （这部分您的原始代码是正确的，保持不变）
                engine = SimulationEngine(self.config, scenario_config)
                scenario_results = engine.run_simulation()
                all_results[scenario_name] = scenario_results
                
                logger.info(f"[SCENARIO SUCCESS] Scenario '{scenario_name}' completed.")
                
            except Exception as e:
                logger.error(f"[SCENARIO FAILED] Scenario '{scenario_name}' failed: {e}")
                import traceback
                logger.error(f"Traceback: {traceback.format_exc()}")
        # --- ▲▲▲ 修改结束 ▲▲▲ ---
        
        return all_results
    
    def analyze_results(self, all_results):
        """分析所有场景的结果并生成KPIs"""
        logger.info("Analyzing results from all scenarios...")
        # (这部分逻辑可以根据需要进一步实现)
        kpis = {}
        for scenario_name, results_path in all_results.items():
            if results_path and os.path.exists(results_path['timeseries_file']):
                df = pd.read_csv(results_path['timeseries_file'])
                kpis[scenario_name] = {'mean_total_load_kw': df['total_load_kw'].mean()}
        return kpis

    def generate_visualizations(self, all_results, kpis):
        """生成可视化图表"""
        logger.info("Generating visualizations...")
        self.visualizer.plot_all(all_results, kpis)
        if ENHANCED_FEATURES_AVAILABLE:
            enhanced_viz = EnhancedVisualizer(self.config)
            enhanced_viz.plot_all(all_results, kpis)

    def run(self):
        """执行完整仿真流程"""
        try:
            # [修复] 替换所有表情符号为ASCII字符
            logger.info("[START] Starting BDWPT Simulation Platform")
            
            if ENHANCED_FEATURES_AVAILABLE:
                logger.info("Running pre-simulation model validation...")
                validator = ModelValidator(self.config)
                validation_report = validator.run_comprehensive_validation()
                if not validation_report:
                     logger.warning("Model validation not available")
                elif validation_report and validation_report.get('overall_score', 0) < 0.7:
                    logger.warning("[WARNING] Model validation score is low. Consider reviewing parameters.")
                logger.info("Continuing with simulation...")
            
            # 主要仿真流程
            self.initialize()
            all_results = self.run_all_scenarios()
            
            if all_results:
                kpis = self.analyze_results(all_results)
                self.generate_visualizations(all_results, kpis)
                
                logger.info("\n" + "="*80)
                logger.info("[COMPLETE] SIMULATION COMPLETED SUCCESSFULLY!")
                logger.info(f"[INFO] Results available in: {self.config.output_dir}")
                logger.info(f"[INFO] Visualizations in: {self.config.figures_dir}")
                logger.info("="*80)
            else:
                logger.warning("[FAILED] No scenarios completed successfully!")
            
        except Exception as e:
            logger.error(f"[FATAL ERROR] Simulation failed: {e}")
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