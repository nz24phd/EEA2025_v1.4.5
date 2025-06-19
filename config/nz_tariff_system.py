# config/nz_tariff_system.py
"""
新西兰电价机制实现
基于新西兰实际电价结构，包含分时电价、季节调整、V2G激励等
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, date
from typing import Dict, Tuple, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class Season(Enum):
    """新西兰季节（南半球）"""
    SUMMER = "summer"    # 12-2月
    AUTUMN = "autumn"    # 3-5月
    WINTER = "winter"    # 6-8月
    SPRING = "spring"    # 9-11月

class TariffType(Enum):
    """电价类型"""
    RESIDENTIAL = "residential"
    COMMERCIAL = "commercial"
    INDUSTRIAL = "industrial"

@dataclass
class GridCondition:
    """电网状态条件"""
    demand_level: float         # 需求水平 (0-1)
    renewable_ratio: float      # 可再生能源占比 (0-1)
    voltage_stability: float    # 电压稳定性 (0-1)
    frequency_deviation: float  # 频率偏差 (Hz)

class NZTariffSystem:
    """新西兰电价系统"""
    
    def __init__(self, region: str = "wellington"):
        """
        初始化新西兰电价系统
        
        Args:
            region: 地区名称（默认惠灵顿）
        """
        self.region = region.lower()
        self.base_rates = self._load_regional_base_rates()
        self.tariff_structure = self._initialize_tariff_structure()
        
        # V2G市场参数
        self.v2g_market = self._initialize_v2g_market()
        
        # 实时电价历史（用于趋势分析）
        self.price_history = []
        
        logger.info(f"NZ tariff system initialized for {region} region")
    
    def _load_regional_base_rates(self) -> Dict[str, float]:
        """
        加载地区基础电价（基于2024年新西兰实际电价）
        
        Returns:
            Dict: 各地区基础电价 (cents/kWh)
        """
        regional_rates = {
            "wellington": {
                "residential_base": 28.5,    # c/kWh
                "commercial_base": 24.2,
                "industrial_base": 18.8,
                "transmission_charge": 3.2,   # 输电费
                "distribution_charge": 8.5,   # 配电费
                "ea_levy": 0.15,             # 电力局征费
                "gst_rate": 0.15             # GST 15%
            },
            "auckland": {
                "residential_base": 29.8,
                "commercial_base": 25.1,
                "industrial_base": 19.5,
                "transmission_charge": 3.8,
                "distribution_charge": 9.2,
                "ea_levy": 0.15,
                "gst_rate": 0.15
            },
            "christchurch": {
                "residential_base": 27.2,
                "commercial_base": 23.5,
                "industrial_base": 17.9,
                "transmission_charge": 2.8,
                "distribution_charge": 7.8,
                "ea_levy": 0.15,
                "gst_rate": 0.15
            }
        }
        
        return regional_rates.get(self.region, regional_rates["wellington"])
    
    def _initialize_tariff_structure(self) -> Dict:
        """
        初始化电价结构
        
        Returns:
            Dict: 电价结构配置
        """
        return {
            # 时段电价倍数
            "time_of_use": {
                "peak_multiplier": 1.45,      # 高峰倍数
                "shoulder_multiplier": 1.10,   # 肩峰倍数
                "off_peak_multiplier": 0.75,   # 低峰倍数
                "super_off_peak_multiplier": 0.65  # 超低峰倍数
            },
            
            # 时段定义（24小时制）
            "time_periods": {
                "weekday": {
                    "super_off_peak": [(0, 6)],           # 00:00-06:00
                    "off_peak": [(6, 7), (10, 17), (21, 24)],  # 其他低峰时段
                    "shoulder": [(17, 18), (20, 21)],     # 肩峰
                    "peak": [(7, 10), (18, 20)]           # 高峰：7-10, 18-20
                },
                "weekend": {
                    "super_off_peak": [(0, 8)],           # 周末早晨
                    "off_peak": [(8, 11), (14, 18), (21, 24)],
                    "shoulder": [(18, 21)],               # 周末晚餐时间
                    "peak": [(11, 14)]                    # 周末中午
                }
            },
            
            # 季节调整因子
            "seasonal_adjustment": {
                Season.SUMMER: 0.95,    # 夏季电价稍低
                Season.AUTUMN: 0.98,    # 秋季基准
                Season.WINTER: 1.20,    # 冬季取暖高峰
                Season.SPRING: 1.02     # 春季略高于秋季
            },
            
            # 需求电价（容量费）
            "demand_charges": {
                "threshold_kw": 50,               # 需求电价起始阈值
                "rate_nzd_per_kw_month": 15.50,   # NZD/kW/月
                "measurement_period_min": 30      # 30分钟需求测量
            },
            
            # 功率因数调整
            "power_factor_adjustment": {
                "threshold": 0.85,      # 功率因数阈值
                "penalty_rate": 0.10,   # 10%惩罚
                "bonus_rate": 0.02      # 2%奖励（>0.95时）
            }
        }
    
    def _initialize_v2g_market(self) -> Dict:
        """
        初始化V2G市场参数
        
        Returns:
            Dict: V2G市场配置
        """
        return {
            # V2G回购电价
            "buyback_rates": {
                "base_rate_ratio": 0.65,        # 基础回购比例（相对零售价）
                "grid_service_premium": 0.15,    # 电网服务溢价
                "frequency_service_rate": 45.0,  # 频率调节服务费 (NZD/MWh)
                "voltage_support_rate": 25.0,    # 电压支撑服务费 (NZD/MWh)
                "peak_shaving_bonus": 0.25       # 削峰填谷奖励倍数
            },
            
            # 激励机制
            "incentives": {
                "high_demand_multiplier": 1.5,   # 高需求时期倍数
                "critical_demand_multiplier": 2.0, # 紧急需求时期
                "renewable_surplus_bonus": 0.10,  # 可再生能源过剩奖励
                "grid_stability_bonus": 0.08      # 电网稳定性奖励
            },
            
            # 市场准入条件
            "participation_requirements": {
                "min_capacity_kwh": 10,          # 最小电池容量
                "min_available_soc": 0.30,       # 最小可用SOC
                "response_time_seconds": 2,      # 响应时间要求
                "availability_hours_per_day": 4  # 每日最小可用时间
            }
        }
    
    def get_current_tariff(self, 
                          timestamp: datetime,
                          tariff_type: TariffType = TariffType.RESIDENTIAL,
                          grid_condition: Optional[GridCondition] = None) -> Tuple[float, Dict]:
        """
        获取当前时刻的电价
        
        Args:
            timestamp: 时间戳
            tariff_type: 电价类型
            grid_condition: 电网状态（可选）
            
        Returns:
            Tuple[float, Dict]: (电价cents/kWh, 详细分解)
        """
        # 1. 基础电价
        base_rate = self._get_base_rate(tariff_type)
        
        # 2. 时段调整
        time_multiplier = self._get_time_multiplier(timestamp)
        
        # 3. 季节调整
        seasonal_factor = self._get_seasonal_factor(timestamp)
        
        # 4. 电网状态调整
        grid_factor = self._get_grid_adjustment(grid_condition) if grid_condition else 1.0
        
        # 5. 计算最终电价
        energy_rate = base_rate * time_multiplier * seasonal_factor * grid_factor
        
        # 6. 添加固定费用
        transmission_rate = self.base_rates["transmission_charge"]
        distribution_rate = self.base_rates["distribution_charge"]
        ea_levy = self.base_rates["ea_levy"]
        
        # 7. 总电价（不含GST）
        total_rate_ex_gst = energy_rate + transmission_rate + distribution_rate + ea_levy
        
        # 8. 添加GST
        gst_amount = total_rate_ex_gst * self.base_rates["gst_rate"]
        final_rate = total_rate_ex_gst + gst_amount
        
        # 详细分解
        breakdown = {
            "energy_rate": energy_rate,
            "transmission_charge": transmission_rate,
            "distribution_charge": distribution_rate,
            "ea_levy": ea_levy,
            "subtotal_ex_gst": total_rate_ex_gst,
            "gst": gst_amount,
            "total_inc_gst": final_rate,
            "time_period": self._get_time_period(timestamp),
            "season": self._get_season(timestamp),
            "multipliers": {
                "time": time_multiplier,
                "seasonal": seasonal_factor,
                "grid": grid_factor
            }
        }
        
        # 记录价格历史
        self.price_history.append({
            "timestamp": timestamp,
            "rate_cents_kwh": final_rate,
            "period": breakdown["time_period"],
            "season": breakdown["season"]
        })
        
        return final_rate, breakdown
    
    def calculate_v2g_rate(self,
                          timestamp: datetime,
                          grid_condition: GridCondition,
                          service_type: str = "energy") -> Tuple[float, Dict]:
        """
        计算V2G回购电价
        
        Args:
            timestamp: 时间戳
            grid_condition: 电网状态
            service_type: 服务类型 ("energy", "frequency", "voltage", "peak_shaving")
            
        Returns:
            Tuple[float, Dict]: (V2G电价cents/kWh, 详细分解)
        """
        # 获取当前零售电价
        retail_rate, _ = self.get_current_tariff(timestamp, TariffType.RESIDENTIAL, grid_condition)
        
        # 基础V2G回购率
        base_buyback_ratio = self.v2g_market["buyback_rates"]["base_rate_ratio"]
        base_v2g_rate = retail_rate * base_buyback_ratio
        
        # 服务类型加成
        service_premiums = {
            "energy": 0.0,  # 纯能量交易无额外加成
            "frequency": self.v2g_market["buyback_rates"]["frequency_service_rate"] / 10,  # 转换为cents/kWh
            "voltage": self.v2g_market["buyback_rates"]["voltage_support_rate"] / 10,
            "peak_shaving": retail_rate * self.v2g_market["buyback_rates"]["peak_shaving_bonus"]
        }
        
        service_premium = service_premiums.get(service_type, 0.0)
        
        # 电网状态激励
        demand_incentive = self._calculate_demand_incentive(grid_condition)
        stability_incentive = self._calculate_stability_incentive(grid_condition)
        renewable_incentive = self._calculate_renewable_incentive(grid_condition)
        
        # 总V2G电价
        total_v2g_rate = (base_v2g_rate + service_premium + 
                         demand_incentive + stability_incentive + renewable_incentive)
        
        # 确保V2G电价不超过零售电价的80%
        max_v2g_rate = retail_rate * 0.80
        final_v2g_rate = min(total_v2g_rate, max_v2g_rate)
        
        breakdown = {
            "base_buyback_rate": base_v2g_rate,
            "service_premium": service_premium,
            "demand_incentive": demand_incentive,
            "stability_incentive": stability_incentive,
            "renewable_incentive": renewable_incentive,
            "total_before_cap": total_v2g_rate,
            "rate_cap": max_v2g_rate,
            "final_v2g_rate": final_v2g_rate,
            "retail_reference": retail_rate
        }
        
        return final_v2g_rate, breakdown
    
    def _get_base_rate(self, tariff_type: TariffType) -> float:
        """获取基础电价"""
        rate_map = {
            TariffType.RESIDENTIAL: "residential_base",
            TariffType.COMMERCIAL: "commercial_base", 
            TariffType.INDUSTRIAL: "industrial_base"
        }
        return self.base_rates[rate_map[tariff_type]]
    
    def _get_time_multiplier(self, timestamp: datetime) -> float:
        """获取时段电价倍数"""
        time_period = self._get_time_period(timestamp)
        return self.tariff_structure["time_of_use"][f"{time_period}_multiplier"]
    
    def _get_time_period(self, timestamp: datetime) -> str:
        """确定时段类型"""
        hour = timestamp.hour
        is_weekend = timestamp.weekday() >= 5
        
        periods = self.tariff_structure["time_periods"]["weekend" if is_weekend else "weekday"]
        
        for period, time_ranges in periods.items():
            for start, end in time_ranges:
                if start <= hour < end:
                    return period
        
        return "off_peak"  # 默认低峰
    
    def _get_seasonal_factor(self, timestamp: datetime) -> float:
        """获取季节调整因子"""
        season = self._get_season(timestamp)
        return self.tariff_structure["seasonal_adjustment"][season]
    
    def _get_season(self, timestamp: datetime) -> Season:
        """确定季节（南半球）"""
        month = timestamp.month
        
        if month in [12, 1, 2]:
            return Season.SUMMER
        elif month in [3, 4, 5]:
            return Season.AUTUMN
        elif month in [6, 7, 8]:
            return Season.WINTER
        else:
            return Season.SPRING
    
    def _get_grid_adjustment(self, grid_condition: GridCondition) -> float:
        """根据电网状态计算调整因子"""
        # 需求水平影响
        if grid_condition.demand_level > 0.9:
            demand_factor = 1.15  # 高需求时电价上涨
        elif grid_condition.demand_level < 0.3:
            demand_factor = 0.90  # 低需求时电价下降
        else:
            demand_factor = 1.0
        
        # 可再生能源占比影响
        if grid_condition.renewable_ratio > 0.8:
            renewable_factor = 0.95  # 高可再生占比时电价下降
        else:
            renewable_factor = 1.0
        
        # 电网稳定性影响
        if grid_condition.voltage_stability < 0.8 or abs(grid_condition.frequency_deviation) > 0.2:
            stability_factor = 1.05  # 不稳定时电价上涨
        else:
            stability_factor = 1.0
        
        return demand_factor * renewable_factor * stability_factor
    
    def _calculate_demand_incentive(self, grid_condition: GridCondition) -> float:
        """计算需求激励"""
        if grid_condition.demand_level > 0.9:
            return 8.0  # 高需求时高激励
        elif grid_condition.demand_level > 0.8:
            return 4.0  # 中等需求时中等激励
        else:
            return 0.0
    
    def _calculate_stability_incentive(self, grid_condition: GridCondition) -> float:
        """计算稳定性激励"""
        voltage_score = max(0, 1 - abs(1 - grid_condition.voltage_stability))
        frequency_score = max(0, 1 - abs(grid_condition.frequency_deviation) / 0.5)
        
        stability_score = (voltage_score + frequency_score) / 2
        
        if stability_score < 0.8:
            return 3.0 * (1 - stability_score)  # 不稳定时提供更多激励
        else:
            return 0.0
    
    def _calculate_renewable_incentive(self, grid_condition: GridCondition) -> float:
        """计算可再生能源激励"""
        if grid_condition.renewable_ratio > 0.8:
            # 可再生能源过剩时鼓励充电（负激励表示更便宜的充电）
            return -2.0
        else:
            return 0.0
    
    def get_daily_price_profile(self, date_obj: date, tariff_type: TariffType = TariffType.RESIDENTIAL) -> pd.DataFrame:
        """
        获取一天的电价曲线
        
        Args:
            date_obj: 日期
            tariff_type: 电价类型
            
        Returns:
            DataFrame: 24小时电价数据
        """
        hours = range(24)
        price_data = []
        
        for hour in hours:
            timestamp = datetime.combine(date_obj, datetime.min.time().replace(hour=hour))
            rate, breakdown = self.get_current_tariff(timestamp, tariff_type)
            
            price_data.append({
                "hour": hour,
                "rate_cents_kwh": rate,
                "time_period": breakdown["time_period"],
                "season": breakdown["season"].value,
                "energy_component": breakdown["energy_rate"],
                "network_charges": breakdown["transmission_charge"] + breakdown["distribution_charge"]
            })
        
        return pd.DataFrame(price_data)
    
    def analyze_price_volatility(self, days: int = 30) -> Dict:
        """
        分析电价波动性
        
        Args:
            days: 分析天数
            
        Returns:
            Dict: 波动性分析结果
        """
        if len(self.price_history) < days * 24:
            return {"error": "Insufficient price history data"}
        
        recent_prices = [record["rate_cents_kwh"] for record in self.price_history[-days * 24:]]
        
        analysis = {
            "mean_price": np.mean(recent_prices),
            "std_price": np.std(recent_prices),
            "min_price": np.min(recent_prices),
            "max_price": np.max(recent_prices),
            "price_range": np.max(recent_prices) - np.min(recent_prices),
            "coefficient_of_variation": np.std(recent_prices) / np.mean(recent_prices),
            "volatility_classification": self._classify_volatility(np.std(recent_prices) / np.mean(recent_prices))
        }
        
        return analysis
    
    def _classify_volatility(self, cv: float) -> str:
        """分类波动性水平"""
        if cv < 0.1:
            return "low"
        elif cv < 0.2:
            return "moderate"
        elif cv < 0.3:
            return "high"
        else:
            return "very_high"
    
    def export_tariff_schedule(self, filepath: str):
        """
        导出电价时刻表
        
        Args:
            filepath: 导出文件路径
        """
        # 创建一个典型工作日和周末的电价时刻表
        schedule_data = []
        
        for day_type in ["weekday", "weekend"]:
            for hour in range(24):
                # 创建示例时间戳
                base_date = datetime(2024, 6, 3 if day_type == "weekday" else 1)  # 周一或周六
                timestamp = base_date.replace(hour=hour)
                
                rate, breakdown = self.get_current_tariff(timestamp)
                
                schedule_data.append({
                    "day_type": day_type,
                    "hour": hour,
                    "time_period": breakdown["time_period"],
                    "rate_cents_kwh": rate,
                    "energy_component": breakdown["energy_rate"],
                    "network_charges": breakdown["transmission_charge"] + breakdown["distribution_charge"]
                })
        
        df = pd.DataFrame(schedule_data)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported tariff schedule to {filepath}")
    
    def validate_tariff_system(self) -> Dict[str, bool]:
        """
        验证电价系统的合理性
        
        Returns:
            Dict: 验证结果
        """
        validation_results = {}
        
        # 测试不同时段的电价
        test_date = datetime(2024, 6, 15, 12, 0)  # 工作日中午
        
        # 验证电价范围合理性
        peak_rate, _ = self.get_current_tariff(test_date.replace(hour=8))  # 高峰
        off_peak_rate, _ = self.get_current_tariff(test_date.replace(hour=2))  # 低峰
        
        validation_results["price_range_reasonable"] = 15 <= off_peak_rate <= 50 and 20 <= peak_rate <= 60
        validation_results["peak_higher_than_off_peak"] = peak_rate > off_peak_rate
        
        # 验证V2G电价合理性
        grid_condition = GridCondition(0.8, 0.6, 0.9, 0.1)
        v2g_rate, _ = self.calculate_v2g_rate(test_date, grid_condition)
        retail_rate, _ = self.get_current_tariff(test_date)
        
        validation_results["v2g_rate_reasonable"] = 0.4 * retail_rate <= v2g_rate <= 0.8 * retail_rate
        
        # 验证季节调整
        winter_rate, _ = self.get_current_tariff(datetime(2024, 7, 15, 12, 0))
        summer_rate, _ = self.get_current_tariff(datetime(2024, 1, 15, 12, 0))
        
        validation_results["winter_higher_than_summer"] = winter_rate > summer_rate
        
        # 验证GST计算
        _, breakdown = self.get_current_tariff(test_date)
        expected_gst = breakdown["subtotal_ex_gst"] * 0.15
        validation_results["gst_calculation_correct"] = abs(breakdown["gst"] - expected_gst) < 0.01
        
        # 整体验证
        validation_results["overall_valid"] = all(validation_results.values())
        
        if validation_results["overall_valid"]:
            logger.info("NZ tariff system validation passed")
        else:
            failed_checks = [k for k, v in validation_results.items() if not v]
            logger.warning(f"NZ tariff system validation failed: {failed_checks}")
        
        return validation_results