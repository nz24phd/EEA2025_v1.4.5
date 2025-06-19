# visualizations/enhanced_visualizations.py
"""
增强可视化模块
提供车流量图、电网G2V/V2G图、电压图对比、拓扑热力图等高级可视化功能
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as patches
import seaborn as sns
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import os

logger = logging.getLogger(__name__)

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class EnhancedVisualizer:
    """增强型可视化器，提供多维度分析图表"""
    
    def __init__(self, config, output_dir: str):
        """
        初始化增强可视化器
        
        Args:
            config: 仿真配置对象
            output_dir: 输出目录
        """
        self.config = config
        self.output_dir = output_dir
        
        # 设置颜色主题
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#FFB30F',
            'info': '#5D737E',
            'g2v': '#4CAF50',     # 绿色 - 网对车
            'v2g': '#FF5722',     # 橙红色 - 车对网
            'baseline': '#9E9E9E'  # 灰色 - 基准线
        }
        
        # 设置样式
        sns.set_style("whitegrid")
        plt.style.use('default')  # 使用默认样式替代可能不存在的样式
        
        logger.info("Enhanced visualizer initialized")
    
    def plot_traffic_flow_heatmap(self, all_results: Dict, save_path: str = None) -> str:
        """
        绘制交通流量热力图
        
        Args:
            all_results: 所有场景的仿真结果
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        # 创建子图布局
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        axes = axes.flatten()
        
        scenarios = [
            'Weekday Peak_0%', 'Weekday Peak_15%', 'Weekday Peak_40%',
            'Weekend Peak_0%', 'Weekend Peak_15%', 'Weekend Peak_40%'
        ]
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            
            if scenario in all_results:
                # 提取车辆分布数据
                df = all_results[scenario]['timeseries']
                vehicle_matrix = self._create_vehicle_distribution_matrix(df)
                
                # 绘制热力图
                im = ax.imshow(vehicle_matrix, cmap='YlOrRd', aspect='auto', interpolation='nearest')
                
                # 设置坐标轴
                ax.set_title(f'{scenario}\n车辆分布热力图', fontsize=12, fontweight='bold')
                ax.set_xlabel('时间 (小时)', fontsize=10)
                ax.set_ylabel('BDWPT节点', fontsize=10)
                
                # 设置刻度
                hours = np.arange(0, 24, 4)
                hour_indices = [h * 4 for h in hours]  # 假设15分钟间隔
                ax.set_xticks(hour_indices)
                ax.set_xticklabels(hours)
                
                node_labels = [str(node) for node in self.config.grid_params['bdwpt_nodes']]
                ax.set_yticks(range(len(node_labels)))
                ax.set_yticklabels(node_labels)
                
                # 添加颜色条
                cbar = plt.colorbar(im, ax=ax, shrink=0.8)
                cbar.set_label('车辆数量', fontsize=9)
                
            else:
                ax.text(0.5, 0.5, f'无数据\n{scenario}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                ax.set_title(scenario, fontsize=12)
        
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = f"{self.output_dir}/traffic_flow_heatmap.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Traffic flow heatmap saved to {save_path}")
        return save_path
    
    def plot_g2v_v2g_power_comparison(self, all_results: Dict, save_path: str = None) -> str:
        """
        绘制G2V/V2G功率对比图
        
        Args:
            all_results: 所有场景的仿真结果
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        
        scenario_groups = [
            ('Weekday Peak', ['Weekday Peak_15%', 'Weekday Peak_40%']),
            ('Weekday Off-Peak', ['Weekday Off-Peak_15%', 'Weekday Off-Peak_40%']),
            ('Weekend', ['Weekend_15%', 'Weekend_40%'])
        ]
        
        for group_idx, (group_name, scenarios) in enumerate(scenario_groups):
            for scenario_idx, scenario in enumerate(scenarios):
                ax = axes[group_idx, scenario_idx]
                
                if scenario in all_results:
                    df = all_results[scenario]['timeseries']
                    
                    # 提取时间和功率数据
                    times = pd.to_datetime(df['timestamp'])
                    hours = times.dt.hour + times.dt.minute / 60
                    
                    g2v_power = df['bdwpt_charging_kw'].fillna(0)
                    v2g_power = df['bdwpt_discharging_kw'].fillna(0)
                    
                    # 绘制面积图
                    ax.fill_between(hours, 0, g2v_power, alpha=0.7, 
                                   color=self.colors['g2v'], label='G2V (充电)')
                    ax.fill_between(hours, 0, -v2g_power, alpha=0.7,
                                   color=self.colors['v2g'], label='V2G (放电)')
                    
                    # 绘制净功率线
                    net_power = g2v_power - v2g_power
                    ax.plot(hours, net_power, color='black', linewidth=2, 
                           linestyle='--', alpha=0.8, label='净功率')
                    
                    # 添加零线
                    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                    
                    # 设置标题和标签
                    penetration = scenario.split('_')[-1]
                    ax.set_title(f'{group_name}\nBDWPT渗透率: {penetration}', 
                               fontsize=11, fontweight='bold')
                    ax.set_ylabel('功率 (kW)', fontsize=10)
                    ax.set_xlim(0, 24)
                    ax.grid(True, alpha=0.3)
                    
                    # 添加图例（只在第一个图中显示）
                    if group_idx == 0 and scenario_idx == 0:
                        ax.legend(loc='upper right', fontsize=9)
                
                else:
                    ax.text(0.5, 0.5, f'无数据\n{scenario}', ha='center', va='center',
                           transform=ax.transAxes, fontsize=11)
                    ax.set_title(scenario, fontsize=11)
        
        # 设置底部子图的x轴标签
        for ax in axes[-1, :]:
            ax.set_xlabel('时间 (小时)', fontsize=10)
            ax.set_xticks(range(0, 25, 4))
        
        plt.suptitle('G2V vs V2G 功率交换模式对比', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = f"{self.output_dir}/g2v_v2g_power_comparison.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"G2V/V2G power comparison saved to {save_path}")
        return save_path
    
    def plot_voltage_profile_comparison(self, all_results: Dict, save_path: str = None) -> str:
        """
        绘制电压曲线对比图
        
        Args:
            all_results: 所有场景的仿真结果
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        # 选择关键节点进行对比
        critical_buses = [632, 633, 671, 675, 680]  # 代表性节点
        
        fig, axes = plt.subplots(len(critical_buses), 1, figsize=(14, 3*len(critical_buses)))
        if len(critical_buses) == 1:
            axes = [axes]
        
        for bus_idx, bus in enumerate(critical_buses):
            ax = axes[bus_idx]
            
            # 绘制不同渗透率的电压曲线
            scenarios_to_plot = ['Weekday Peak_0%', 'Weekday Peak_15%', 'Weekday Peak_40%']
            colors = [self.colors['baseline'], self.colors['primary'], self.colors['accent']]
            linestyles = ['-', '--', '-.']
            
            for scenario, color, linestyle in zip(scenarios_to_plot, colors, linestyles):
                if scenario in all_results:
                    df = all_results[scenario]['timeseries']
                    voltage_col = f'voltage_bus_{bus}'
                    
                    if voltage_col in df.columns:
                        times = pd.to_datetime(df['timestamp'])
                        hours = times.dt.hour + times.dt.minute / 60
                        voltages = df[voltage_col]
                        
                        penetration = scenario.split('_')[-1]
                        label = f"{penetration} BDWPT" if penetration != "0%" else "基准场景"
                        
                        ax.plot(hours, voltages, color=color, linewidth=2, 
                               linestyle=linestyle, label=label, alpha=0.8)
            
            # 添加电压限制线
            ax.axhline(y=1.05, color='red', linestyle=':', alpha=0.7, label='上限 (1.05 p.u.)')
            ax.axhline(y=0.95, color='red', linestyle=':', alpha=0.7, label='下限 (0.95 p.u.)')
            ax.axhline(y=1.00, color='gray', linestyle='-', alpha=0.5, label='额定电压')
            
            # 设置图表属性
            ax.set_title(f'节点 {bus} 电压曲线', fontsize=12, fontweight='bold')
            ax.set_ylabel('电压 (p.u.)', fontsize=10)
            ax.set_xlim(0, 24)
            ax.set_ylim(0.93, 1.07)
            ax.grid(True, alpha=0.3)
            
            # 添加图例（只在第一个图中显示）
            if bus_idx == 0:
                ax.legend(loc='upper right', fontsize=9, ncol=2)
        
        # 设置最后一个子图的x轴标签
        axes[-1].set_xlabel('时间 (小时)', fontsize=10)
        axes[-1].set_xticks(range(0, 25, 4))
        
        plt.suptitle('关键节点电压曲线对比分析', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = f"{self.output_dir}/voltage_profile_comparison.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Voltage profile comparison saved to {save_path}")
        return save_path
    
    def plot_grid_topology_heatmap(self, all_results: Dict, save_path: str = None) -> str:
        """
        绘制电网拓扑热力图
        
        Args:
            all_results: 所有场景的仿真结果
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        scenarios = [
            'Weekday Peak_0%', 'Weekday Peak_15%', 'Weekday Peak_40%',
            'Weekend Peak_0%', 'Weekend Peak_15%', 'Weekend Peak_40%'
        ]
        
        # IEEE 13节点系统的拓扑位置
        node_positions = self._get_ieee13_node_positions()
        
        for idx, scenario in enumerate(scenarios):
            ax = axes[idx]
            
            if scenario in all_results:
                df = all_results[scenario]['timeseries']
                
                # 计算各节点的平均BDWPT功率
                node_powers = self._calculate_average_node_powers(df)
                
                # 绘制网络拓扑
                self._draw_network_topology(ax, node_positions, node_powers, scenario)
                
            else:
                ax.text(0.5, 0.5, f'无数据\n{scenario}', ha='center', va='center',
                       transform=ax.transAxes, fontsize=12)
            
            ax.set_title(scenario, fontsize=12, fontweight='bold')
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('IEEE 13节点系统 BDWPT功率分布拓扑图', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        
        # 保存图片
        if save_path is None:
            save_path = f"{self.output_dir}/grid_topology_heatmap.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Grid topology heatmap saved to {save_path}")
        return save_path
    
    def plot_comprehensive_kpi_dashboard(self, kpis: Dict, save_path: str = None) -> str:
        """
        绘制综合KPI仪表板
        
        Args:
            kpis: KPI数据字典
            save_path: 保存路径
            
        Returns:
            str: 保存的文件路径
        """
        fig = plt.figure(figsize=(20, 14))
        
        # 创建网格布局
        from matplotlib import gridspec
        gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. 峰值削减效果 (左上)
        ax1 = fig.add_subplot(gs[0, :2])
        self._plot_peak_reduction_bar(ax1, kpis)
        
        # 2. V2G能量贡献 (右上)
        ax2 = fig.add_subplot(gs[0, 2:])
        self._plot_v2g_energy_pie(ax2, kpis)
        
        # 3. 损耗变化分析 (左中)
        ax3 = fig.add_subplot(gs[1, :2])
        self._plot_loss_change_analysis(ax3, kpis)
        
        # 4. 电压质量改善 (右中)
        ax4 = fig.add_subplot(gs[1, 2:])
        self._plot_voltage_quality_radar(ax4, kpis)
        
        # 5. 场景对比矩阵 (下部)
        ax5 = fig.add_subplot(gs[2, :])
        self._plot_scenario_comparison_matrix(ax5, kpis)
        
        plt.suptitle('BDWPT系统性能关键指标仪表板', fontsize=18, fontweight='bold', y=0.98)
        
        # 保存图片
        if save_path is None:
            save_path = f"{self.output_dir}/comprehensive_kpi_dashboard.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Comprehensive KPI dashboard saved to {save_path}")
        return save_path
    
    def _create_vehicle_distribution_matrix(self, df: pd.DataFrame) -> np.ndarray:
        """创建车辆分布矩阵"""
        nodes = self.config.grid_params['bdwpt_nodes']
        time_steps = len(df)
        
        # 创建矩阵：行为节点，列为时间步
        matrix = np.zeros((len(nodes), time_steps))
        
        for i, node in enumerate(nodes):
            # 查找包含车辆数据的列
            vehicle_cols = [col for col in df.columns if f'node_{node}' in col and 'vehicles' in col]
            if vehicle_cols:
                # 使用第一个匹配的列，或者合并多个列
                if len(vehicle_cols) == 1:
                    matrix[i, :] = df[vehicle_cols[0]].fillna(0)
                else:
                    # 合并G2V和V2G车辆数
                    g2v_col = [col for col in vehicle_cols if 'G2V' in col]
                    v2g_col = [col for col in vehicle_cols if 'V2G' in col]
                    idle_col = [col for col in vehicle_cols if 'idle' in col]
                    
                    total_vehicles = np.zeros(time_steps)
                    if g2v_col:
                        total_vehicles += df[g2v_col[0]].fillna(0)
                    if v2g_col:
                        total_vehicles += df[v2g_col[0]].fillna(0)
                    if idle_col:
                        total_vehicles += df[idle_col[0]].fillna(0)
                    
                    matrix[i, :] = total_vehicles
            else:
                # 如果没有直接的车辆数据，根据功率估算
                power_col = f'bdwpt_node_{node}_kw'
                if power_col in df.columns:
                    powers = df[power_col].fillna(0)
                    # 假设每辆车平均功率25kW来估算车辆数
                    estimated_vehicles = np.abs(powers) / 25
                    matrix[i, :] = estimated_vehicles
        
        return matrix
    
    def _get_ieee13_node_positions(self) -> Dict[int, Tuple[float, float]]:
        """获取IEEE 13节点系统的拓扑位置"""
        # 基于IEEE 13节点标准拓扑的相对位置
        positions = {
            632: (0, 0),      # 变电站（源点）
            633: (2, 0),      # 主馈线第一节点
            634: (4, 0),      # 主馈线延伸
            645: (2, -2),     # 分支1
            646: (4, -2),     # 分支1延伸
            671: (1, 2),      # 分支2
            675: (1, 4),      # 分支2延伸
            680: (6, 0),      # 最远端节点
        }
        return positions
    
    def _calculate_average_node_powers(self, df: pd.DataFrame) -> Dict[int, float]:
        """计算各节点的平均BDWPT功率"""
        node_powers = {}
        
        for node in self.config.grid_params['bdwpt_nodes']:
            power_col = f'bdwpt_node_{node}_kw'
            if power_col in df.columns:
                node_powers[node] = df[power_col].mean()
            else:
                node_powers[node] = 0.0
        
        return node_powers
    
    def _draw_network_topology(self, ax, positions: Dict, powers: Dict, scenario: str):
        """绘制网络拓扑图（不使用networkx）"""
        # 手动绘制网络拓扑
        
        # 定义连接关系（基于IEEE 13节点拓扑）
        edges = [
            (632, 633), (633, 634), (634, 680),  # 主馈线
            (632, 645), (645, 646),              # 分支1
            (632, 671), (671, 675)               # 分支2
        ]
        
        # 绘制边
        for edge in edges:
            node1, node2 = edge
            if node1 in positions and node2 in positions:
                x1, y1 = positions[node1]
                x2, y2 = positions[node2]
                ax.plot([x1, x2], [y1, y2], 'gray', linewidth=2, alpha=0.6)
        
        # 绘制节点（大小和颜色反映功率）
        max_abs_power = max(abs(p) for p in powers.values()) if powers.values() else 1
        
        for node, (x, y) in positions.items():
            power = powers.get(node, 0)
            
            # 节点大小基于功率大小
            size = 300 + (abs(power) / max_abs_power) * 1000 if max_abs_power > 0 else 300
            
            # 节点颜色基于功率方向
            if power > 0:
                color = self.colors['g2v']  # 充电为绿色
                alpha = 0.7
            elif power < 0:
                color = self.colors['v2g']  # 放电为红色
                alpha = 0.7
            else:
                color = self.colors['baseline']  # 无功率为灰色
                alpha = 0.5
            
            # 绘制节点
            ax.scatter(x, y, s=size, c=color, alpha=alpha, edgecolors='black', linewidth=2)
            
            # 添加节点标签
            ax.annotate(f'N{node}\n{power:.1f}kW', (x, y), 
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, ha='left', va='bottom',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        # 设置坐标轴
        ax.set_xlim(-1, 7)
        ax.set_ylim(-3, 5)
        ax.set_xlabel('相对位置 X', fontsize=10)
        ax.set_ylabel('相对位置 Y', fontsize=10)
    
    def _plot_peak_reduction_bar(self, ax, kpis: Dict):
        """绘制峰值削减柱状图"""
        scenarios = []
        peak_reductions = []
        
        for scenario, metrics in kpis.items():
            if '0%' not in scenario:  # 排除基准场景
                scenarios.append(scenario.replace('_', '\n'))
                peak_reductions.append(metrics.get('peak_reduction_kw', 0))
        
        if scenarios:
            bars = ax.bar(scenarios, peak_reductions, color=self.colors['primary'], alpha=0.7)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        ax.set_title('峰值负荷削减效果', fontsize=12, fontweight='bold')
        ax.set_ylabel('削减功率 (kW)', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_v2g_energy_pie(self, ax, kpis: Dict):
        """绘制V2G能量贡献饼图"""
        v2g_energies = []
        labels = []
        
        for scenario, metrics in kpis.items():
            if '40%' in scenario:  # 只显示高渗透率场景
                v2g_energy = metrics.get('energy_from_v2g_kwh', 0)
                if v2g_energy > 0:
                    v2g_energies.append(v2g_energy)
                    labels.append(scenario.replace('_40%', ''))
        
        if v2g_energies:
            colors = [self.colors['primary'], self.colors['secondary'], self.colors['accent']][:len(v2g_energies)]
            wedges, texts, autotexts = ax.pie(v2g_energies, labels=labels, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
            
            # 美化文本
            for autotext in autotexts:
                autotext.set_color('white')
                autotext.set_fontweight('bold')
        
        ax.set_title('V2G能量贡献分布\n(40%渗透率场景)', fontsize=12, fontweight='bold')
    
    def _plot_loss_change_analysis(self, ax, kpis: Dict):
        """绘制损耗变化分析"""
        scenarios = []
        loss_changes = []
        colors = []
        
        for scenario, metrics in kpis.items():
            if '0%' not in scenario:
                scenarios.append(scenario.replace('_', '\n'))
                loss_change = metrics.get('loss_reduction_kwh', 0)
                loss_changes.append(loss_change)
                
                # 损耗增加为红色，减少为绿色
                colors.append(self.colors['success'] if loss_change > 0 else self.colors['warning'])
        
        if scenarios:
            bars = ax.bar(scenarios, loss_changes, color=colors, alpha=0.7)
            
            # 添加零线
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 添加数值标签
            for bar in bars:
                height = bar.get_height()
                va = 'bottom' if height >= 0 else 'top'
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va=va, fontsize=9)
        
        ax.set_title('系统损耗变化分析', fontsize=12, fontweight='bold')
        ax.set_ylabel('损耗变化 (kWh)', fontsize=10)
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
    
    def _plot_voltage_quality_radar(self, ax, kpis: Dict):
        """绘制电压质量雷达图"""
        # 这里是一个简化的示例，实际实现需要更多的电压质量指标
        categories = ['电压偏差', '电压波动', '电压不平衡', '谐波畸变', '电压稳定性']
        
        # 模拟一些电压质量指标数据
        baseline_scores = [0.6, 0.7, 0.8, 0.9, 0.7]
        improved_scores = [0.8, 0.85, 0.9, 0.92, 0.85]
        
        angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        baseline_scores += baseline_scores[:1]
        improved_scores += improved_scores[:1]
        
        ax.plot(angles, baseline_scores, 'o-', linewidth=2, label='基准场景', color=self.colors['baseline'])
        ax.fill(angles, baseline_scores, alpha=0.25, color=self.colors['baseline'])
        
        ax.plot(angles, improved_scores, 'o-', linewidth=2, label='BDWPT改善', color=self.colors['primary'])
        ax.fill(angles, improved_scores, alpha=0.25, color=self.colors['primary'])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title('电压质量指标雷达图', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        ax.grid(True)
    
    def _plot_scenario_comparison_matrix(self, ax, kpis: Dict):
        """绘制场景对比矩阵"""
        # 准备数据
        scenarios = [s for s in kpis.keys() if '0%' not in s]
        metrics = ['peak_reduction_kw', 'loss_reduction_kwh', 'energy_from_v2g_kwh']
        metric_labels = ['峰值削减(kW)', '损耗减少(kWh)', 'V2G能量(kWh)']
        
        # 创建数据矩阵
        data_matrix = np.zeros((len(scenarios), len(metrics)))
        
        for i, scenario in enumerate(scenarios):
            for j, metric in enumerate(metrics):
                data_matrix[i, j] = kpis[scenario].get(metric, 0)
        
        # 标准化数据用于颜色映射
        data_normalized = np.zeros_like(data_matrix)
        for j in range(len(metrics)):
            col_data = data_matrix[:, j]
            if col_data.max() - col_data.min() > 1e-8:  # 避免除零
                data_normalized[:, j] = (col_data - col_data.min()) / (col_data.max() - col_data.min())
        
        # 绘制热力图
        im = ax.imshow(data_normalized, cmap='RdYlBu_r', aspect='auto')
        
        # 设置刻度和标签
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, fontsize=10)
        ax.set_yticks(range(len(scenarios)))
        ax.set_yticklabels([s.replace('_', ' ') for s in scenarios], fontsize=9)
        
        # 添加数值文本
        for i in range(len(scenarios)):
            for j in range(len(metrics)):
                value = data_matrix[i, j]
                text_color = 'white' if data_normalized[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontweight='bold')
        
        ax.set_title('场景性能对比矩阵', fontsize=12, fontweight='bold')
        
        # 添加颜色条
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=20)
        cbar.set_label('标准化性能值', fontsize=9)
    
    def generate_all_visualizations(self, all_results: Dict, kpis: Dict) -> List[str]:
        """
        生成所有可视化图表
        
        Args:
            all_results: 所有场景结果
            kpis: KPI数据
            
        Returns:
            List[str]: 生成的文件路径列表
        """
        logger.info("Generating comprehensive visualization suite...")
        
        generated_files = []
        
        try:
            # 确保输出目录存在
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 1. 交通流量热力图
            path1 = self.plot_traffic_flow_heatmap(all_results)
            generated_files.append(path1)
            
            # 2. G2V/V2G功率对比
            path2 = self.plot_g2v_v2g_power_comparison(all_results)
            generated_files.append(path2)
            
            # 3. 电压曲线对比
            path3 = self.plot_voltage_profile_comparison(all_results)
            generated_files.append(path3)
            
            # 4. 电网拓扑热力图
            path4 = self.plot_grid_topology_heatmap(all_results)
            generated_files.append(path4)
            
            # 5. 综合KPI仪表板
            path5 = self.plot_comprehensive_kpi_dashboard(kpis)
            generated_files.append(path5)
            
            logger.info(f"Successfully generated {len(generated_files)} visualization files")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
            raise
        
        return generated_files