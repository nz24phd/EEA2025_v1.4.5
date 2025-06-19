# traffic_model/enhanced_vehicle_movement.py
"""
增强的车辆移动模式实现
基于新西兰城市交通特征和IEEE 13节点配电网拓扑
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)

class EnhancedVehicleMovement:
    """增强的车辆移动仿真器，提供更真实的车辆分布和移动模式"""
    
    def __init__(self, road_network, config):
        """
        初始化增强车辆移动仿真器
        
        Args:
            road_network: 道路网络数据
            config: 仿真配置对象
        """
        self.road_network = road_network
        self.config = config
        self.vehicle_positions = {}
        
        # 构建新西兰城市特征的距离和时间矩阵
        self.distance_matrix = self._build_nz_distance_matrix()
        self.travel_time_matrix = self._build_travel_time_matrix()
        
        # 交通流量分配权重（基于惠灵顿城市结构）
        self.node_weights = self._calculate_node_weights()
        
        logger.info("Enhanced vehicle movement system initialized with NZ urban characteristics")
    
    def _build_nz_distance_matrix(self) -> pd.DataFrame:
        """
        构建基于新西兰城市特征的节点间距离矩阵
        
        Returns:
            DataFrame: 节点间距离矩阵(km)
        """
        nodes = self.config.grid_params['bdwpt_nodes']
        n = len(nodes)
        distances = np.zeros((n, n))
        
        # 新西兰城市配电网特征：节点间距离相对较短且集中
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j:
                    distances[i][j] = self._get_nz_realistic_distance(node_i, node_j)
        
        df = pd.DataFrame(distances, index=nodes, columns=nodes)
        logger.debug(f"Built distance matrix for {len(nodes)} nodes")
        return df
    
    def _get_nz_realistic_distance(self, node1: int, node2: int) -> float:
        """
        计算基于新西兰城市配电网的真实节点间距离
        
        Args:
            node1, node2: 节点编号
            
        Returns:
            float: 距离(km)
        """
        # 基于IEEE 13节点拓扑和新西兰城市特征的距离映射
        # 惠灵顿等城市的配电网节点间距离通常为0.3-1.8km
        distance_map = {
            # 主干线距离
            (632, 633): 1.2,   # 变电站到主要分支
            (633, 634): 0.9,   # 主干线延伸
            (634, 680): 1.1,   # 到最远端
            
            # 分支线距离  
            (632, 645): 0.8,   # 分支1
            (645, 646): 0.7,   # 分支1延伸
            (632, 671): 1.0,   # 分支2
            (671, 675): 0.6,   # 分支2延伸
            
            # 跨分支距离（通过中转）
            (633, 645): 1.4,
            (634, 646): 1.3,
            (645, 671): 1.6,
            (646, 675): 1.8,
            (680, 675): 2.1    # 最远端距离
        }
        
        # 标准化节点对
        key = tuple(sorted([node1, node2]))
        
        if key in distance_map:
            return distance_map[key]
        else:
            # 对于未定义的节点对，基于拓扑估算
            return self._estimate_distance_by_topology(node1, node2)
    
    def _estimate_distance_by_topology(self, node1: int, node2: int) -> float:
        """
        基于拓扑结构估算距离
        
        Args:
            node1, node2: 节点编号
            
        Returns:
            float: 估算距离(km)
        """
        # 简化的拓扑距离计算
        # 基于节点编号的相对位置估算
        base_distance = abs(node1 - node2) * 0.2  # 基础距离
        
        # 新西兰城市配电网距离范围：0.3-2.0km
        estimated = np.clip(base_distance + np.random.uniform(0.3, 0.8), 0.3, 2.0)
        
        return round(estimated, 2)
    
    def _build_travel_time_matrix(self) -> pd.DataFrame:
        """
        构建行程时间矩阵
        
        Returns:
            DataFrame: 节点间行程时间矩阵(分钟)
        """
        nodes = self.config.grid_params['bdwpt_nodes']
        
        # 新西兰城市平均行驶速度
        avg_speed_kmh = {
            'peak_hours': 25,      # 高峰时段较慢
            'off_peak': 35,        # 非高峰时段
            'weekend': 30          # 周末中等速度
        }
        
        # 使用非高峰速度作为基准
        base_speed = avg_speed_kmh['off_peak']
        
        travel_times = self.distance_matrix * 60 / base_speed  # 转换为分钟
        
        logger.debug("Built travel time matrix with NZ urban speed characteristics")
        return travel_times
    
    def _calculate_node_weights(self) -> Dict[int, float]:
        """
        计算各节点的交通流量权重（基于新西兰城市特征）
        
        Returns:
            Dict: 节点权重字典
        """
        # 基于新西兰城市结构的节点重要性权重
        # 考虑商业区、住宅区、交通枢纽的分布
        weights = {
            632: 0.20,  # 变电站附近：中央商务区
            633: 0.18,  # 主干线：主要商业走廊
            634: 0.15,  # 延伸区：混合用地
            645: 0.12,  # 分支1：住宅区
            646: 0.10,  # 分支1延伸：郊区住宅
            671: 0.13,  # 分支2：教育/医疗区
            675: 0.08,  # 分支2延伸：低密度住宅
            680: 0.04   # 最远端：工业/仓储区
        }
        
        # 标准化权重
        total_weight = sum(weights.values())
        normalized_weights = {k: v/total_weight for k, v in weights.items()}
        
        logger.info("Calculated node weights based on NZ urban characteristics")
        return normalized_weights
    
    def update_positions_enhanced(self, vehicles: List[Dict], trips: pd.DataFrame, 
                                  current_time_minutes: int) -> Dict[int, List[Dict]]:
        """
        增强的车辆位置更新算法
        
        Args:
            vehicles: 车辆列表
            trips: 行程数据框
            current_time_minutes: 当前时间（分钟）
            
        Returns:
            Dict: 各节点的车辆分布
        """
        # 重置车辆状态
        for vehicle in vehicles:
            if vehicle['status'] == 'driving':
                vehicle['status'] = 'parked'
        
        # 获取当前活跃行程
        active_trips = trips[
            (trips['departure_time'] <= current_time_minutes) &
            (trips['arrival_time'] > current_time_minutes)
        ].copy()
        
        # 处理每个活跃行程
        for _, trip in active_trips.iterrows():
            vehicle_id = trip['vehicle_id']
            
            if 0 <= vehicle_id < len(vehicles):
                self._update_vehicle_trip_status(vehicles[vehicle_id], trip, current_time_minutes)
        
        # 处理刚完成的行程
        finished_trips = trips[trips['arrival_time'] == current_time_minutes]
        for _, trip in finished_trips.iterrows():
            vehicle_id = trip['vehicle_id']
            if 0 <= vehicle_id < len(vehicles):
                vehicles[vehicle_id]['status'] = 'parked'
                vehicles[vehicle_id]['location'] = trip['destination']
                logger.debug(f"Vehicle {vehicle_id} completed trip to node {trip['destination']}")
        
        # 更新车辆位置分布
        vehicles_by_node = self._distribute_vehicles_by_node(vehicles, active_trips)
        
        return vehicles_by_node
    
    def _update_vehicle_trip_status(self, vehicle: Dict, trip: pd.Series, current_time: int):
        """
        更新单个车辆的行程状态
        
        Args:
            vehicle: 车辆对象
            trip: 行程信息
            current_time: 当前时间
        """
        # 计算行程进度
        trip_duration = trip['arrival_time'] - trip['departure_time']
        elapsed_time = current_time - trip['departure_time']
        progress = elapsed_time / trip_duration if trip_duration > 0 else 0
        
        # 更新车辆状态和位置
        vehicle['status'] = 'driving'
        
        # 根据行程进度确定位置
        if progress < 0.3:
            # 行程前30%：在起点附近
            vehicle['location'] = trip['origin']
            vehicle['trip_stage'] = 'departing'
        elif progress < 0.7:
            # 行程中段40%：在路途中
            vehicle['location'] = self._interpolate_location(trip['origin'], trip['destination'], progress)
            vehicle['trip_stage'] = 'en_route'
        else:
            # 行程后30%：接近目的地
            vehicle['location'] = trip['destination']
            vehicle['trip_stage'] = 'arriving'
        
        # 更新能耗
        if vehicle['type'] == 'EV':
            self._update_vehicle_energy_consumption(vehicle, trip, progress)
    
    def _interpolate_location(self, origin: int, destination: int, progress: float) -> int:
        """
        基于行程进度插值计算当前位置
        
        Args:
            origin: 起点节点
            destination: 终点节点  
            progress: 行程进度(0-1)
            
        Returns:
            int: 当前位置节点
        """
        # 简化处理：在行程中段时随机分配到路径上的节点
        if progress < 0.5:
            return origin
        else:
            return destination
    
    def _update_vehicle_energy_consumption(self, vehicle: Dict, trip: pd.Series, progress: float):
        """
        更新车辆能耗（基于新西兰城市驾驶条件）
        
        Args:
            vehicle: 车辆对象
            trip: 行程信息
            progress: 行程进度
        """
        if vehicle.get('battery_capacity_kwh', 0) <= 0:
            return
            
        # 获取行程总距离
        origin, destination = trip['origin'], trip['destination']
        
        # 安全检查：确保节点存在于距离矩阵中
        if origin not in self.distance_matrix.index or destination not in self.distance_matrix.columns:
            logger.warning(f"Distance not found for trip {origin} -> {destination}")
            return
            
        total_distance = self.distance_matrix.loc[origin, destination]
        
        # 计算本时间步的行程距离
        time_step_hours = self.config.time_step_minutes / 60
        
        # 基于进度计算距离增量
        previous_progress = max(0, progress - 0.1)  # 上一时间步进度
        distance_increment = total_distance * (progress - previous_progress)
        
        # 新西兰城市驾驶能耗因子
        base_consumption = self.config.ev_params['energy_consumption_kwh_per_km']
        
        # 城市驾驶修正因子
        city_factors = {
            'traffic_density': 1.15,    # 交通密度影响
            'stop_start': 1.20,         # 走走停停
            'hills': 1.10,              # 新西兰城市地形起伏
            'weather': 1.05             # 气候影响
        }
        
        total_factor = np.prod(list(city_factors.values()))
        actual_consumption = base_consumption * total_factor
        
        # 计算能耗
        energy_used = distance_increment * actual_consumption
        soc_decrease = energy_used / vehicle['battery_capacity_kwh']
        
        # 更新SOC
        vehicle['current_soc'] = max(0.05, vehicle['current_soc'] - soc_decrease)
        
        # 记录详细信息用于调试
        if 'energy_log' not in vehicle:
            vehicle['energy_log'] = []
        
        vehicle['energy_log'].append({
            'distance_km': distance_increment,
            'energy_kwh': energy_used,
            'soc_after': vehicle['current_soc']
        })
    
    def _distribute_vehicles_by_node(self, vehicles: List[Dict], active_trips: pd.DataFrame) -> Dict[int, List[Dict]]:
        """
        按节点分布车辆
        
        Args:
            vehicles: 车辆列表
            active_trips: 活跃行程
            
        Returns:
            Dict: 各节点的车辆分布
        """
        vehicles_by_node = {node: [] for node in self.config.grid_params['bdwpt_nodes']}
        
        for vehicle in vehicles:
            location = vehicle.get('location')
            
            # 确保位置是有效的BDWPT节点
            if location in self.config.grid_params['bdwpt_nodes']:
                vehicles_by_node[location].append(vehicle)
            elif vehicle['status'] == 'driving':
                # 对于行驶中的车辆，根据权重分配到最可能的节点
                likely_node = self._assign_driving_vehicle_node(vehicle, active_trips)
                vehicles_by_node[likely_node].append(vehicle)
        
        # 记录分布统计
        total_vehicles = sum(len(v_list) for v_list in vehicles_by_node.values())
        logger.debug(f"Distributed {total_vehicles} vehicles across {len(vehicles_by_node)} nodes")
        
        return vehicles_by_node
    
    def _assign_driving_vehicle_node(self, vehicle: Dict, active_trips: pd.DataFrame) -> int:
        """
        为行驶中的车辆分配最可能的节点位置
        
        Args:
            vehicle: 车辆对象
            active_trips: 活跃行程数据
            
        Returns:
            int: 分配的节点编号
        """
        vehicle_id = vehicle['id']
        
        # 查找该车辆的当前行程
        vehicle_trip = active_trips[active_trips['vehicle_id'] == vehicle_id]
        
        if not vehicle_trip.empty:
            trip = vehicle_trip.iloc[0]
            
            # 根据行程阶段分配节点
            if vehicle.get('trip_stage') == 'departing':
                return trip['origin']
            elif vehicle.get('trip_stage') == 'arriving':
                return trip['destination']
            else:
                # 行程中段：根据权重随机选择
                return np.random.choice(
                    list(self.node_weights.keys()),
                    p=list(self.node_weights.values())
                )
        else:
            # 没有找到对应行程，随机分配
            return np.random.choice(self.config.grid_params['bdwpt_nodes'])
    
    def get_traffic_density_by_node(self, current_time_minutes: int) -> Dict[int, float]:
        """
        获取各节点的交通密度
        
        Args:
            current_time_minutes: 当前时间
            
        Returns:
            Dict: 各节点交通密度
        """
        hour = (current_time_minutes // 60) % 24
        
        # 新西兰城市交通密度模式
        base_density = self._get_base_traffic_density(hour)
        
        # 应用节点权重
        density_by_node = {}
        for node, weight in self.node_weights.items():
            density_by_node[node] = base_density * weight
        
        return density_by_node
    
    def _get_base_traffic_density(self, hour: int) -> float:
        """
        获取基础交通密度（基于新西兰城市模式）
        
        Args:
            hour: 小时(0-23)
            
        Returns:
            float: 基础交通密度
        """
        # 新西兰城市交通模式：双峰明显
        density_profile = np.array([
            0.1, 0.05, 0.03, 0.03, 0.05, 0.15,  # 0-5点：夜间低峰
            0.4, 0.8, 1.0, 0.7, 0.5, 0.6,       # 6-11点：晨峰
            0.7, 0.6, 0.5, 0.6, 0.7, 0.9,       # 12-17点：下午
            1.0, 0.8, 0.6, 0.4, 0.25, 0.15      # 18-23点：晚峰
        ])
        
        return density_profile[hour]
    
    def generate_movement_statistics(self, vehicles: List[Dict]) -> Dict:
        """
        生成车辆移动统计信息
        
        Args:
            vehicles: 车辆列表
            
        Returns:
            Dict: 统计信息
        """
        stats = {
            'total_vehicles': len(vehicles),
            'driving_vehicles': len([v for v in vehicles if v['status'] == 'driving']),
            'parked_vehicles': len([v for v in vehicles if v['status'] == 'parked']),
            'average_soc': np.mean([v.get('current_soc', 0) for v in vehicles if v['type'] == 'EV']),
            'low_soc_vehicles': len([v for v in vehicles if v.get('current_soc', 1) < 0.2]),
            'node_distribution': {}
        }
        
        # 节点分布统计
        for node in self.config.grid_params['bdwpt_nodes']:
            stats['node_distribution'][node] = len([
                v for v in vehicles if v.get('location') == node
            ])
        
        return stats
    
    def validate_movement_patterns(self) -> Dict[str, bool]:
        """
        验证车辆移动模式的合理性
        
        Returns:
            Dict: 验证结果
        """
        validation = {
            'distance_matrix_symmetric': self._check_distance_symmetry(),
            'travel_times_realistic': self._check_travel_time_realism(),
            'node_weights_normalized': abs(sum(self.node_weights.values()) - 1.0) < 0.01,
            'nz_characteristics_applied': self._check_nz_characteristics()
        }
        
        all_valid = all(validation.values())
        validation['overall_valid'] = all_valid
        
        if all_valid:
            logger.info("Vehicle movement patterns validation passed")
        else:
            failed_checks = [k for k, v in validation.items() if not v]
            logger.warning(f"Vehicle movement validation failed: {failed_checks}")
        
        return validation
    
    def _check_distance_symmetry(self) -> bool:
        """检查距离矩阵对称性"""
        return np.allclose(self.distance_matrix.values, self.distance_matrix.values.T)
    
    def _check_travel_time_realism(self) -> bool:
        """检查行程时间的合理性"""
        max_time = self.travel_time_matrix.max().max()
        min_time = self.travel_time_matrix[self.travel_time_matrix > 0].min().min()
        
        # 新西兰城市内行程时间应在1-40分钟范围内
        return 1 <= min_time <= 40 and max_time <= 40
    
    def _check_nz_characteristics(self) -> bool:
        """检查是否正确应用了新西兰特征"""
        # 检查距离范围是否符合新西兰城市特征
        distances = self.distance_matrix.values
        non_zero_distances = distances[distances > 0]
        
        # 新西兰城市配电网节点间距离：0.3-2.0km
        distance_range_ok = np.all((non_zero_distances >= 0.3) & (non_zero_distances <= 2.0))
        
        # 检查节点权重是否反映城市结构
        cbd_nodes = [632, 633]  # CBD节点应有较高权重
        suburban_nodes = [675, 680]  # 郊区节点应有较低权重
        
        cbd_weight = sum(self.node_weights[node] for node in cbd_nodes)
        suburban_weight = sum(self.node_weights[node] for node in suburban_nodes)
        
        weight_distribution_ok = cbd_weight > suburban_weight
        
        return distance_range_ok and weight_distribution_ok