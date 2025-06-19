# power_grid_model/bdwpt_efficiency_model.py
"""
动态BDWPT效率模型
基于文献中的90-93%效率范围，考虑多种影响因素的精细化建模
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class ThermalState(Enum):
    """热状态枚举"""
    COLD = "cold"
    NORMAL = "normal" 
    WARM = "warm"
    HOT = "hot"
    OVERHEATING = "overheating"

@dataclass
class BDWPTCharacteristics:
    """BDWPT系统特性参数"""
    coil_alignment_quality: float = 0.90    # 线圈对准质量 (0.8-0.98)
    air_gap_mm: float = 100                  # 气隙距离 (80-150mm)
    operating_frequency_khz: float = 85      # 工作频率 (85kHz)
    coil_efficiency: float = 0.95            # 线圈本身效率
    power_electronics_efficiency: float = 0.96  # 电力电子效率
    thermal_state: ThermalState = ThermalState.NORMAL
    magnetic_coupling: float = 0.20          # 磁耦合系数 (0.15-0.25)

class DynamicBDWPTEfficiency:
    """动态BDWPT效率计算器"""
    
    def __init__(self, config):
        """
        初始化动态效率模型
        
        Args:
            config: 仿真配置对象
        """
        self.config = config
        
        # 基础效率参数（基于文献数据）
        self.base_efficiency = 0.91  # 理想条件下的基础效率
        self.efficiency_history = []
        
        # 环境参数
        self.ambient_temp_c = 20
        self.humidity_percent = 60
        
        # 系统老化因子（模拟长期使用影响）
        self.aging_factor = 1.0
        
        logger.info("Dynamic BDWPT efficiency model initialized")
    
    def calculate_efficiency(self, 
                           power_kw: float, 
                           characteristics: BDWPTCharacteristics,
                           ambient_temp_c: float = 20,
                           grid_frequency_hz: float = 50) -> Tuple[float, Dict]:
        """
        计算动态BDWPT效率
        
        Args:
            power_kw: 传输功率 (kW)
            characteristics: BDWPT系统特性
            ambient_temp_c: 环境温度 (°C)
            grid_frequency_hz: 电网频率 (Hz)
            
        Returns:
            Tuple[float, Dict]: (效率值, 详细分析)
        """
        # 1. 功率相关效率
        power_efficiency = self._calculate_power_efficiency(power_kw)
        
        # 2. 线圈对准效率
        alignment_efficiency = self._calculate_alignment_efficiency(characteristics)
        
        # 3. 气隙效率
        airgap_efficiency = self._calculate_airgap_efficiency(characteristics)
        
        # 4. 温度效率
        thermal_efficiency = self._calculate_thermal_efficiency(
            ambient_temp_c, characteristics.thermal_state
        )
        
        # 5. 频率稳定性效率
        frequency_efficiency = self._calculate_frequency_efficiency(
            characteristics.operating_frequency_khz, grid_frequency_hz
        )
        
        # 6. 电力电子效率
        electronics_efficiency = characteristics.power_electronics_efficiency
        
        # 7. 磁耦合效率
        coupling_efficiency = self._calculate_coupling_efficiency(characteristics)
        
        # 综合效率计算
        total_efficiency = (
            self.base_efficiency *
            power_efficiency *
            alignment_efficiency *
            airgap_efficiency *
            thermal_efficiency *
            frequency_efficiency *
            electronics_efficiency *
            coupling_efficiency *
            self.aging_factor
        )
        
        # 添加小幅随机波动（模拟实际测量不确定性）
        measurement_noise = np.random.normal(0, 0.005)  # ±0.5% 标准差
        total_efficiency += measurement_noise
        
        # 限制在合理范围内（70-95%）
        total_efficiency = np.clip(total_efficiency, 0.70, 0.95)
        
        # 详细分析
        analysis = {
            'power_factor': power_efficiency,
            'alignment_factor': alignment_efficiency,
            'airgap_factor': airgap_efficiency,
            'thermal_factor': thermal_efficiency,
            'frequency_factor': frequency_efficiency,
            'electronics_factor': electronics_efficiency,
            'coupling_factor': coupling_efficiency,
            'aging_factor': self.aging_factor,
            'final_efficiency': total_efficiency,
            'power_kw': power_kw,
            'ambient_temp_c': ambient_temp_c
        }
        
        # 记录历史数据
        self.efficiency_history.append(analysis)
        
        return total_efficiency, analysis
    
    def _calculate_power_efficiency(self, power_kw: float) -> float:
        """
        计算功率相关的效率因子
        
        Args:
            power_kw: 传输功率
            
        Returns:
            float: 功率效率因子
        """
        # BDWPT系统在额定功率附近效率最高
        rated_power = self.config.bdwpt_params['max_power_kw']
        power_ratio = power_kw / rated_power
        
        if power_ratio < 0.1:
            # 极低功率时效率下降明显
            return 0.80 + power_ratio * 1.5
        elif power_ratio <= 0.8:
            # 正常工作范围，效率较高且稳定
            return 0.98 + 0.02 * np.sin(np.pi * power_ratio)
        else:
            # 过载时效率下降
            return 0.98 - (power_ratio - 0.8) * 0.15
    
    def _calculate_alignment_efficiency(self, characteristics: BDWPTCharacteristics) -> float:
        """
        计算线圈对准效率
        
        Args:
            characteristics: BDWPT系统特性
            
        Returns:
            float: 对准效率因子
        """
        alignment_quality = characteristics.coil_alignment_quality
        
        # 对准质量直接影响磁通耦合
        # 完美对准(1.0)时效率最高，偏移时效率快速下降
        if alignment_quality >= 0.95:
            return 1.0
        elif alignment_quality >= 0.85:
            return 0.95 + 0.05 * (alignment_quality - 0.85) / 0.10
        else:
            # 严重失准时效率显著下降
            return 0.75 + 0.20 * (alignment_quality - 0.80) / 0.05
    
    def _calculate_airgap_efficiency(self, characteristics: BDWPTCharacteristics) -> float:
        """
        计算气隙距离效率
        
        Args:
            characteristics: BDWPT系统特性
            
        Returns:
            float: 气隙效率因子
        """
        air_gap = characteristics.air_gap_mm
        
        # 气隙距离与效率成反比关系
        # 设计气隙通常为100mm，效率曲线基于Maxwell方程
        if air_gap <= 80:
            return 1.0  # 最小安全距离时效率最高
        elif air_gap <= 120:
            # 线性下降区间
            return 1.0 - (air_gap - 80) * 0.003  # 每mm下降0.3%
        else:
            # 大气隙时效率快速下降
            return 0.88 - (air_gap - 120) * 0.005
    
    def _calculate_thermal_efficiency(self, 
                                    ambient_temp_c: float, 
                                    thermal_state: ThermalState) -> float:
        """
        计算温度相关效率
        
        Args:
            ambient_temp_c: 环境温度
            thermal_state: 系统热状态
            
        Returns:
            float: 温度效率因子
        """
        # 基础温度效率（基于半导体器件特性）
        if 15 <= ambient_temp_c <= 25:
            base_temp_factor = 1.0  # 理想温度范围
        elif 5 <= ambient_temp_c < 15:
            # 低温时器件效率略降
            base_temp_factor = 0.98 + (ambient_temp_c - 5) * 0.002
        elif 25 < ambient_temp_c <= 40:
            # 高温时效率下降
            base_temp_factor = 1.0 - (ambient_temp_c - 25) * 0.008
        else:
            # 极端温度
            base_temp_factor = max(0.85, 1.0 - abs(ambient_temp_c - 20) * 0.01)
        
        # 系统热状态影响
        thermal_state_factors = {
            ThermalState.COLD: 0.95,        # 冷启动效率稍低
            ThermalState.NORMAL: 1.0,       # 正常工作温度
            ThermalState.WARM: 0.98,        # 温度较高但可接受
            ThermalState.HOT: 0.90,         # 高温保护开始作用
            ThermalState.OVERHEATING: 0.75   # 过热保护限制功率
        }
        
        state_factor = thermal_state_factors[thermal_state]
        
        return base_temp_factor * state_factor
    
    def _calculate_frequency_efficiency(self, 
                                      operating_freq_khz: float, 
                                      grid_freq_hz: float) -> float:
        """
        计算频率稳定性效率
        
        Args:
            operating_freq_khz: BDWPT工作频率
            grid_freq_hz: 电网频率
            
        Returns:
            float: 频率效率因子
        """
        # 1. BDWPT工作频率稳定性
        nominal_freq = 85  # kHz, ISM频段
        freq_deviation = abs(operating_freq_khz - nominal_freq) / nominal_freq
        
        if freq_deviation < 0.01:  # ±1%内
            freq_stability_factor = 1.0
        elif freq_deviation < 0.05:  # ±5%内
            freq_stability_factor = 0.98 - freq_deviation * 10
        else:
            freq_stability_factor = 0.90  # 频率偏移过大
        
        # 2. 电网频率质量影响
        nominal_grid_freq = 50  # Hz for NZ
        grid_freq_deviation = abs(grid_freq_hz - nominal_grid_freq) / nominal_grid_freq
        
        if grid_freq_deviation < 0.005:  # ±0.5%内，电网质量好
            grid_quality_factor = 1.0
        else:
            grid_quality_factor = max(0.95, 1.0 - grid_freq_deviation * 20)
        
        return freq_stability_factor * grid_quality_factor
    
    def _calculate_coupling_efficiency(self, characteristics: BDWPTCharacteristics) -> float:
        """
        计算磁耦合效率
        
        Args:
            characteristics: BDWPT系统特性
            
        Returns:
            float: 磁耦合效率因子
        """
        coupling_coefficient = characteristics.magnetic_coupling
        
        # 磁耦合系数与传输效率的关系
        # k = 0.2 时可达到较好的效率
        optimal_coupling = 0.20
        
        if coupling_coefficient >= optimal_coupling:
            return 1.0
        else:
            # 耦合不足时效率下降
            coupling_ratio = coupling_coefficient / optimal_coupling
            return 0.85 + 0.15 * coupling_ratio
    
    def update_thermal_state(self, 
                           current_power_kw: float, 
                           duration_minutes: float,
                           ambient_temp_c: float) -> ThermalState:
        """
        更新系统热状态
        
        Args:
            current_power_kw: 当前功率
            duration_minutes: 持续时间
            ambient_temp_c: 环境温度
            
        Returns:
            ThermalState: 更新后的热状态
        """
        # 简化的热模型：基于功率和时间估算温升
        power_density = current_power_kw / self.config.bdwpt_params['max_power_kw']
        
        # 温升计算（简化模型）
        temp_rise = power_density * duration_minutes * 0.1 + ambient_temp_c
        
        if temp_rise < 30:
            return ThermalState.COLD
        elif temp_rise < 60:
            return ThermalState.NORMAL
        elif temp_rise < 80:
            return ThermalState.WARM
        elif temp_rise < 100:
            return ThermalState.HOT
        else:
            return ThermalState.OVERHEATING
    
    def get_efficiency_statistics(self) -> Dict:
        """
        获取效率统计信息
        
        Returns:
            Dict: 效率统计数据
        """
        if not self.efficiency_history:
            return {}
        
        efficiencies = [record['final_efficiency'] for record in self.efficiency_history]
        
        stats = {
            'mean_efficiency': np.mean(efficiencies),
            'std_efficiency': np.std(efficiencies),
            'min_efficiency': np.min(efficiencies),
            'max_efficiency': np.max(efficiencies),
            'total_records': len(self.efficiency_history),
            'efficiency_trend': self._calculate_efficiency_trend()
        }
        
        return stats
    
    def _calculate_efficiency_trend(self) -> str:
        """
        计算效率趋势
        
        Returns:
            str: 趋势描述
        """
        if len(self.efficiency_history) < 10:
            return "insufficient_data"
        
        recent_efficiencies = [
            record['final_efficiency'] 
            for record in self.efficiency_history[-10:]
        ]
        early_efficiencies = [
            record['final_efficiency'] 
            for record in self.efficiency_history[:10]
        ]
        
        recent_avg = np.mean(recent_efficiencies)
        early_avg = np.mean(early_efficiencies)
        
        if recent_avg > early_avg + 0.01:
            return "improving"
        elif recent_avg < early_avg - 0.01:
            return "degrading"
        else:
            return "stable"
    
    def validate_efficiency_model(self) -> Dict[str, bool]:
        """
        验证效率模型的合理性
        
        Returns:
            Dict: 验证结果
        """
        validation_results = {}
        
        # 测试不同功率下的效率
        test_powers = [5, 15, 25, 35, 45]  # kW
        test_characteristics = BDWPTCharacteristics()
        
        efficiencies = []
        for power in test_powers:
            eff, _ = self.calculate_efficiency(power, test_characteristics)
            efficiencies.append(eff)
        
        # 验证效率范围
        validation_results['efficiency_in_range'] = all(
            0.70 <= eff <= 0.95 for eff in efficiencies
        )
        
        # 验证效率曲线合理性（在额定功率附近应该最高）
        max_eff_index = np.argmax(efficiencies)
        rated_power_index = 2  # 25kW对应索引
        validation_results['peak_efficiency_reasonable'] = abs(max_eff_index - rated_power_index) <= 1
        
        # 验证温度影响
        cold_eff, _ = self.calculate_efficiency(25, test_characteristics, ambient_temp_c=5)
        hot_eff, _ = self.calculate_efficiency(25, test_characteristics, ambient_temp_c=40)
        validation_results['temperature_effect_correct'] = cold_eff < hot_eff or abs(cold_eff - hot_eff) < 0.05
        
        # 验证气隙影响
        small_gap_char = BDWPTCharacteristics(air_gap_mm=80)
        large_gap_char = BDWPTCharacteristics(air_gap_mm=150)
        
        small_gap_eff, _ = self.calculate_efficiency(25, small_gap_char)
        large_gap_eff, _ = self.calculate_efficiency(25, large_gap_char)
        validation_results['airgap_effect_correct'] = small_gap_eff > large_gap_eff
        
        # 整体验证
        validation_results['overall_valid'] = all(validation_results.values())
        
        if validation_results['overall_valid']:
            logger.info("BDWPT efficiency model validation passed")
        else:
            failed_checks = [k for k, v in validation_results.items() if not v]
            logger.warning(f"BDWPT efficiency model validation failed: {failed_checks}")
        
        return validation_results
    
    def export_efficiency_data(self, filepath: str):
        """
        导出效率历史数据
        
        Args:
            filepath: 导出文件路径
        """
        if not self.efficiency_history:
            logger.warning("No efficiency data to export")
            return
        
        df = pd.DataFrame(self.efficiency_history)
        df.to_csv(filepath, index=False)
        logger.info(f"Exported {len(self.efficiency_history)} efficiency records to {filepath}")