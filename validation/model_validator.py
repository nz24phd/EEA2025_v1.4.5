# validation/model_validator.py
"""
模型验证框架
验证仿真模型参数的合理性，确保与新西兰实际情况匹配
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, date
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """验证结果数据类"""
    parameter: str
    is_valid: bool
    configured_value: float
    reference_value: float
    deviation_percent: float
    recommendation: str
    priority: str  # 'high', 'medium', 'low'

class ModelValidator:
    """模型验证器，验证仿真参数的合理性"""
    
    def __init__(self, config):
        """
        初始化模型验证器
        
        Args:
            config: 仿真配置对象
        """
        self.config = config
        self.validation_results = {}
        
        # 新西兰参考数据
        self.nz_reference_data = self._load_nz_reference_data()
        
        # 验证标准
        self.validation_standards = self._load_validation_standards()
        
        logger.info("Model validator initialized with NZ reference data")
    
    def _load_nz_reference_data(self) -> Dict:
        """
        加载新西兰参考数据
        
        Returns:
            Dict: 新西兰参考数据
        """
        return {
            # 电动车参数（基于新西兰2024年市场数据）
            'ev_parameters': {
                'battery_capacity_kwh': {
                    'mean': 58.5,           # 市场加权平均
                    'std': 18.2,            # 标准差
                    'min': 35,              # 最小值（如Nissan Leaf）
                    'max': 100,             # 最大值（如Tesla Model S）
                    'popular_models': {
                        'Nissan Leaf': 40,
                        'Tesla Model 3': 75,
                        'Hyundai Kona EV': 64,
                        'MG ZS EV': 44,
                        'BMW i3': 42
                    }
                },
                'energy_consumption_kwh_per_km': {
                    'urban': 0.18,          # 城市驾驶
                    'highway': 0.14,        # 高速公路
                    'combined': 0.16,       # 综合工况
                    'winter_factor': 1.25,  # 冬季能耗增加
                    'summer_factor': 0.95   # 夏季能耗减少
                },
                'charging_power_kw': {
                    'home_ac': 7.4,         # 家用交流充电
                    'public_ac': 22,        # 公共交流充电
                    'dc_fast': 50,          # 直流快充
                    'ultra_fast': 150       # 超快充
                },
                'penetration_2024': 0.12,  # 12% EV渗透率
                'growth_rate_annual': 0.35  # 35%年增长率
            },
            
            # 交通模式参数
            'traffic_patterns': {
                'average_trip_distance_km': {
                    'wellington': 15.8,
                    'auckland': 18.2,
                    'christchurch': 14.5,
                    'hamilton': 13.1,
                    'national_average': 15.4
                },
                'trips_per_day': {
                    'urban': 2.8,
                    'suburban': 3.2,
                    'rural': 2.1
                },
                'peak_hours': {
                    'morning_start': 7,
                    'morning_end': 9,
                    'evening_start': 17,
                    'evening_end': 19
                },
                'modal_split': {
                    'private_vehicle': 0.78,
                    'public_transport': 0.12,
                    'walking_cycling': 0.10
                }
            },
            
            # 电网参数
            'grid_parameters': {
                'voltage_levels': {
                    'low_voltage': 0.4,     # kV
                    'medium_voltage': 11,    # kV  
                    'high_voltage': 33      # kV
                },
                'frequency': 50,            # Hz
                'voltage_tolerance': 0.06,  # ±6%
                'power_quality_standards': 'AS/NZS 61000',
                'peak_demand_residential_kw': 8.5,
                'average_load_factor': 0.45
            },
            
            # 电价参数
            'electricity_pricing': {
                'residential_cents_kwh': {
                    'wellington': 28.5,
                    'auckland': 29.8,
                    'christchurch': 27.2,
                    'national_average': 28.5
                },
                'time_of_use_ratio': {
                    'peak_multiplier': 1.45,
                    'off_peak_multiplier': 0.75
                },
                'seasonal_variation': 0.25,  # 25%季节变化
                'carbon_price_nzd_tonne': 85
            },
            
            # BDWPT技术参数
            'bdwpt_technology': {
                'efficiency_range': (0.85, 0.93),
                'power_levels_kw': [20, 30, 50, 100],
                'air_gap_range_mm': (80, 150),
                'operating_frequency_khz': 85,
                'deployment_readiness': 0.7  # 技术成熟度
            }
        }
    
    def _load_validation_standards(self) -> Dict:
        """
        加载验证标准
        
        Returns:
            Dict: 验证标准配置
        """
        return {
            'tolerance_levels': {
                'high_precision': 0.05,     # ±5%
                'medium_precision': 0.15,   # ±15%
                'low_precision': 0.30       # ±30%
            },
            'critical_parameters': [
                'battery_capacity_kwh',
                'energy_consumption_kwh_per_km',
                'average_trip_distance_km',
                'base_voltage_kv',
                'bdwpt_efficiency'
            ],
            'acceptable_ranges': {
                'ev_penetration': (0.05, 0.50),        # 5%-50%
                'trip_distance_km': (5, 30),           # 5-30km
                'charging_power_kw': (3, 200),         # 3-200kW
                'grid_voltage_kv': (0.2, 50),          # 0.2-50kV
                'efficiency': (0.70, 0.98)             # 70%-98%
            }
        }
    
    def validate_ev_parameters(self) -> Dict[str, ValidationResult]:
        """
        验证电动车参数
        
        Returns:
            Dict: EV参数验证结果
        """
        results = {}
        
        # 验证电池容量
        configured_capacity = getattr(self.config, 'default_battery_capacity', 
                                    self.config.ev_params.get('battery_capacity_kwh', 60))
        reference_capacity = self.nz_reference_data['ev_parameters']['battery_capacity_kwh']['mean']
        
        results['battery_capacity'] = self._validate_parameter(
            'battery_capacity_kwh',
            configured_capacity,
            reference_capacity,
            tolerance=0.20,
            recommendation="建议使用58.5kWh作为新西兰EV平均电池容量"
        )
        
        # 验证能耗
        configured_consumption = self.config.ev_params.get('energy_consumption_kwh_per_km', 0.15)
        reference_consumption = self.nz_reference_data['ev_parameters']['energy_consumption_kwh_per_km']['combined']
        
        results['energy_consumption'] = self._validate_parameter(
            'energy_consumption_kwh_per_km',
            configured_consumption,
            reference_consumption,
            tolerance=0.15,
            recommendation="建议使用0.16kWh/km作为新西兰城市综合能耗"
        )
        
        # 验证充电功率
        configured_power = self.config.bdwpt_params.get('charging_power_kw', 50)
        reference_power = 50  # BDWPT典型功率
        
        results['charging_power'] = self._validate_parameter(
            'charging_power_kw',
            configured_power,
            reference_power,
            tolerance=0.10,
            recommendation="50kW是BDWPT的合理功率水平"
        )
        
        # 验证EV渗透率
        configured_penetration = self.config.traffic_params.get('ev_penetration', 0.3)
        reference_penetration = self.nz_reference_data['ev_parameters']['penetration_2024']
        
        results['ev_penetration'] = self._validate_parameter(
            'ev_penetration',
            configured_penetration,
            reference_penetration,
            tolerance=0.50,  # 允许较大差异（未来预测）
            recommendation=f"当前新西兰EV渗透率约{reference_penetration:.1%}，可根据研究目标调整"
        )
        
        self.validation_results['ev_parameters'] = results
        return results
    
    def validate_traffic_patterns(self) -> Dict[str, ValidationResult]:
        """
        验证交通模式参数
        
        Returns:
            Dict: 交通模式验证结果
        """
        results = {}
        
        # 验证平均行程距离
        configured_distance = self.config.traffic_params.get('average_trip_distance_km', 8.5)
        reference_distance = self.nz_reference_data['traffic_patterns']['average_trip_distance_km']['national_average']
        
        results['trip_distance'] = self._validate_parameter(
            'average_trip_distance_km',
            configured_distance,
            reference_distance,
            tolerance=0.20,
            recommendation=f"建议使用{reference_distance}km作为新西兰平均行程距离"
        )
        
        # 验证每日行程次数
        configured_trips = self.config.traffic_params.get('trips_per_vehicle_per_day', 2.5)
        reference_trips = self.nz_reference_data['traffic_patterns']['trips_per_day']['urban']
        
        results['trips_per_day'] = self._validate_parameter(
            'trips_per_vehicle_per_day',
            configured_trips,
            reference_trips,
            tolerance=0.15,
            recommendation=f"建议使用{reference_trips}次/天作为城市地区平均行程次数"
        )
        
        # 验证峰值小时因子
        configured_phf = self.config.traffic_params.get('peak_hour_factor', 1.5)
        reference_phf = 1.3  # 新西兰城市典型值
        
        results['peak_hour_factor'] = self._validate_parameter(
            'peak_hour_factor',
            configured_phf,
            reference_phf,
            tolerance=0.25,
            recommendation="新西兰城市峰值小时因子通常为1.2-1.4"
        )
        
        self.validation_results['traffic_patterns'] = results
        return results
    
    def validate_grid_parameters(self) -> Dict[str, ValidationResult]:
        """
        验证电网参数
        
        Returns:
            Dict: 电网参数验证结果
        """
        results = {}
        
        # 验证基础电压
        configured_voltage = self.config.grid_params.get('base_voltage_kv', 4.16)
        reference_voltage = self.nz_reference_data['grid_parameters']['voltage_levels']['medium_voltage']
        
        # IEEE 13节点系统需要缩放到新西兰标准
        scaled_voltage = configured_voltage * (reference_voltage / 4.16)
        
        results['base_voltage'] = ValidationResult(
            parameter='base_voltage_kv',
            is_valid=abs(scaled_voltage - reference_voltage) / reference_voltage < 0.10,
            configured_value=configured_voltage,
            reference_value=reference_voltage,
            deviation_percent=abs(scaled_voltage - reference_voltage) / reference_voltage * 100,
            recommendation=f"IEEE 13节点系统应缩放至{reference_voltage}kV以匹配新西兰配电网",
            priority='high'
        )
        
        # 验证电压容差
        configured_tolerance = self.config.grid_params.get('voltage_tolerance', 0.05)
        reference_tolerance = self.nz_reference_data['grid_parameters']['voltage_tolerance']
        
        results['voltage_tolerance'] = self._validate_parameter(
            'voltage_tolerance',
            configured_tolerance,
            reference_tolerance,
            tolerance=0.20,
            recommendation=f"新西兰电压容差标准为±{reference_tolerance*100:.0f}%"
        )
        
        # 验证频率
        configured_freq = getattr(self.config, 'grid_frequency', 50)  # 默认50Hz
        reference_freq = self.nz_reference_data['grid_parameters']['frequency']
        
        results['grid_frequency'] = self._validate_parameter(
            'grid_frequency_hz',
            configured_freq,
            reference_freq,
            tolerance=0.01,
            recommendation="新西兰电网频率为50Hz"
        )
        
        self.validation_results['grid_parameters'] = results
        return results
    
    def validate_economic_parameters(self) -> Dict[str, ValidationResult]:
        """
        验证经济参数
        
        Returns:
            Dict: 经济参数验证结果
        """
        results = {}
        
        # 验证基础电价
        configured_rate = getattr(self.config, 'base_electricity_rate', 28.5)
        reference_rate = self.nz_reference_data['electricity_pricing']['residential_cents_kwh']['national_average']
        
        results['electricity_rate'] = self._validate_parameter(
            'base_electricity_rate_cents_kwh',
            configured_rate,
            reference_rate,
            tolerance=0.15,
            recommendation=f"新西兰住宅电价平均为{reference_rate}c/kWh"
        )
        
        # 验证分时电价倍数
        peak_multiplier = getattr(self.config, 'peak_multiplier', 1.4)
        reference_multiplier = self.nz_reference_data['electricity_pricing']['time_of_use_ratio']['peak_multiplier']
        
        results['peak_multiplier'] = self._validate_parameter(
            'peak_electricity_multiplier',
            peak_multiplier,
            reference_multiplier,
            tolerance=0.20,
            recommendation=f"新西兰高峰电价倍数通常为{reference_multiplier}"
        )
        
        self.validation_results['economic_parameters'] = results
        return results
    
    def validate_bdwpt_technology(self) -> Dict[str, ValidationResult]:
        """
        验证BDWPT技术参数
        
        Returns:
            Dict: BDWPT技术验证结果
        """
        results = {}
        
        # 验证效率
        configured_efficiency = self.config.bdwpt_params.get('efficiency', 0.85)
        reference_efficiency_range = self.nz_reference_data['bdwpt_technology']['efficiency_range']
        reference_efficiency = np.mean(reference_efficiency_range)
        
        results['bdwpt_efficiency'] = ValidationResult(
            parameter='bdwpt_efficiency',
            is_valid=reference_efficiency_range[0] <= configured_efficiency <= reference_efficiency_range[1],
            configured_value=configured_efficiency,
            reference_value=reference_efficiency,
            deviation_percent=abs(configured_efficiency - reference_efficiency) / reference_efficiency * 100,
            recommendation=f"BDWPT效率应在{reference_efficiency_range[0]:.0%}-{reference_efficiency_range[1]:.0%}范围内",
            priority='high'
        )
        
        # 验证工作频率
        configured_freq = getattr(self.config.bdwpt_params, 'operating_frequency_khz', 85)
        reference_freq = self.nz_reference_data['bdwpt_technology']['operating_frequency_khz']
        
        results['operating_frequency'] = self._validate_parameter(
            'operating_frequency_khz',
            configured_freq,
            reference_freq,
            tolerance=0.05,
            recommendation=f"BDWPT标准工作频率为{reference_freq}kHz（ISM频段）"
        )
        
        # 验证功率水平
        configured_power = self.config.bdwpt_params.get('max_power_kw', 50)
        reference_powers = self.nz_reference_data['bdwpt_technology']['power_levels_kw']
        
        results['power_level'] = ValidationResult(
            parameter='max_power_kw',
            is_valid=configured_power in reference_powers,
            configured_value=configured_power,
            reference_value=50,  # 常用功率等级
            deviation_percent=0 if configured_power in reference_powers else 20,
            recommendation=f"建议使用标准功率等级：{reference_powers}kW",
            priority='medium'
        )
        
        self.validation_results['bdwpt_technology'] = results
        return results
    
    def _validate_parameter(self, 
                          parameter_name: str,
                          configured_value: float,
                          reference_value: float,
                          tolerance: float = 0.15,
                          recommendation: str = "") -> ValidationResult:
        """
        验证单个参数
        
        Args:
            parameter_name: 参数名称
            configured_value: 配置值
            reference_value: 参考值
            tolerance: 容差
            recommendation: 建议
            
        Returns:
            ValidationResult: 验证结果
        """
        deviation = abs(configured_value - reference_value) / reference_value
        is_valid = deviation <= tolerance
        
        # 确定优先级
        if parameter_name in self.validation_standards['critical_parameters']:
            priority = 'high'
        elif deviation > 0.25:
            priority = 'medium'
        else:
            priority = 'low'
        
        return ValidationResult(
            parameter=parameter_name,
            is_valid=is_valid,
            configured_value=configured_value,
            reference_value=reference_value,
            deviation_percent=deviation * 100,
            recommendation=recommendation,
            priority=priority
        )
    
    def run_comprehensive_validation(self) -> Dict:
        """
        运行全面验证
        
        Returns:
            Dict: 完整验证报告
        """
        logger.info("Running comprehensive model validation...")
        
        # 运行各类验证
        ev_results = self.validate_ev_parameters()
        traffic_results = self.validate_traffic_patterns()
        grid_results = self.validate_grid_parameters()
        economic_results = self.validate_economic_parameters()
        bdwpt_results = self.validate_bdwpt_technology()
        
        # 汇总结果
        all_results = {
            'ev_parameters': ev_results,
            'traffic_patterns': traffic_results,
            'grid_parameters': grid_results,
            'economic_parameters': economic_results,
            'bdwpt_technology': bdwpt_results
        }
        
        # 生成综合报告
        report = self._generate_validation_report(all_results)
        
        logger.info(f"Validation completed. Overall score: {report['overall_score']:.1%}")
        
        return report
    
    def _generate_validation_report(self, all_results: Dict) -> Dict:
        """
        生成验证报告
        
        Args:
            all_results: 所有验证结果
            
        Returns:
            Dict: 验证报告
        """
        total_parameters = 0
        valid_parameters = 0
        critical_issues = []
        recommendations = []
        
        # 统计验证结果
        for category, results in all_results.items():
            for param_name, result in results.items():
                total_parameters += 1
                
                if isinstance(result, ValidationResult):
                    if result.is_valid:
                        valid_parameters += 1
                    elif result.priority == 'high':
                        critical_issues.append({
                            'category': category,
                            'parameter': result.parameter,
                            'issue': f"偏差{result.deviation_percent:.1f}%",
                            'recommendation': result.recommendation
                        })
                    
                    if result.recommendation:
                        recommendations.append({
                            'category': category,
                            'parameter': result.parameter,
                            'priority': result.priority,
                            'recommendation': result.recommendation
                        })
        
        # 计算整体分数
        overall_score = valid_parameters / total_parameters if total_parameters > 0 else 0
        
        # 生成建议优先级排序
        recommendations.sort(key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_score': overall_score,
            'total_parameters': total_parameters,
            'valid_parameters': valid_parameters,
            'validation_grade': self._get_validation_grade(overall_score),
            'critical_issues': critical_issues,
            'recommendations': recommendations[:10],  # 前10个建议
            'detailed_results': all_results,
            'summary_by_category': self._summarize_by_category(all_results),
            'nz_adaptation_status': self._assess_nz_adaptation(all_results)
        }
        
        return report
    
    def _get_validation_grade(self, score: float) -> str:
        """获取验证等级"""
        if score >= 0.90:
            return 'A'
        elif score >= 0.80:
            return 'B'
        elif score >= 0.70:
            return 'C'
        elif score >= 0.60:
            return 'D'
        else:
            return 'F'
    
    def _summarize_by_category(self, all_results: Dict) -> Dict:
        """按类别汇总结果"""
        summary = {}
        
        for category, results in all_results.items():
            valid_count = sum(1 for r in results.values() 
                            if isinstance(r, ValidationResult) and r.is_valid)
            total_count = len(results)
            
            summary[category] = {
                'valid_parameters': valid_count,
                'total_parameters': total_count,
                'validity_rate': valid_count / total_count if total_count > 0 else 0,
                'status': 'good' if valid_count / total_count > 0.8 else 
                         'needs_attention' if valid_count / total_count > 0.6 else 'critical'
            }
        
        return summary
    
    def _assess_nz_adaptation(self, all_results: Dict) -> Dict:
        """评估新西兰本地化适配状态"""
        # 关键的新西兰特征参数
        key_nz_parameters = [
            ('grid_parameters', 'base_voltage'),
            ('grid_parameters', 'grid_frequency'),
            ('traffic_patterns', 'trip_distance'),
            ('economic_parameters', 'electricity_rate'),
            ('ev_parameters', 'energy_consumption')
        ]
        
        adapted_params = 0
        for category, param in key_nz_parameters:
            if category in all_results and param in all_results[category]:
                result = all_results[category][param]
                if isinstance(result, ValidationResult) and result.is_valid:
                    adapted_params += 1
        
        adaptation_score = adapted_params / len(key_nz_parameters)
        
        return {
            'adaptation_score': adaptation_score,
            'adapted_parameters': adapted_params,
            'total_key_parameters': len(key_nz_parameters),
            'status': 'fully_adapted' if adaptation_score > 0.8 else
                     'partially_adapted' if adaptation_score > 0.5 else
                     'needs_adaptation',
            'recommendation': self._get_adaptation_recommendation(adaptation_score)
        }
    
    def _get_adaptation_recommendation(self, score: float) -> str:
        """获取适配建议"""
        if score > 0.8:
            return "模型已良好适配新西兰环境，可以进行可靠的仿真分析"
        elif score > 0.5:
            return "模型部分适配新西兰环境，建议调整关键参数以提高准确性"
        else:
            return "模型需要显著调整以适配新西兰环境，建议优先处理关键参数"
    
    def export_validation_report(self, filepath: str):
        """
        导出验证报告
        
        Args:
            filepath: 导出文件路径
        """
        report = self.run_comprehensive_validation()
        
        # 转换ValidationResult对象为字典以便JSON序列化
        serializable_report = self._make_serializable(report)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(serializable_report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Validation report exported to {filepath}")
    
    def _make_serializable(self, obj):
        """将对象转换为可序列化格式"""
        if isinstance(obj, ValidationResult):
            return {
                'parameter': obj.parameter,
                'is_valid': obj.is_valid,
                'configured_value': obj.configured_value,
                'reference_value': obj.reference_value,
                'deviation_percent': obj.deviation_percent,
                'recommendation': obj.recommendation,
                'priority': obj.priority
            }
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def get_validation_summary(self) -> str:
        """
        获取验证结果摘要
        
        Returns:
            str: 验证摘要文本
        """
        report = self.run_comprehensive_validation()
        
        summary = f"""
模型验证报告摘要
================

整体评估：{report['validation_grade']} 级
验证通过率：{report['overall_score']:.1%} ({report['valid_parameters']}/{report['total_parameters']})

新西兰适配状态：{report['nz_adaptation_status']['status']}
适配评分：{report['nz_adaptation_status']['adaptation_score']:.1%}

关键问题数量：{len(report['critical_issues'])}
优先建议数量：{len([r for r in report['recommendations'] if r['priority'] == 'high'])}

各类别状态：
"""
        
        for category, status in report['summary_by_category'].items():
            summary += f"- {category}: {status['validity_rate']:.1%} ({status['status']})\n"
        
        if report['critical_issues']:
            summary += "\n关键问题：\n"
            for issue in report['critical_issues'][:3]:  # 显示前3个
                summary += f"- {issue['category']}.{issue['parameter']}: {issue['issue']}\n"
        
        return summary