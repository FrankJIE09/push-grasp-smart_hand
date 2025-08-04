import numpy as np
from scipy.spatial import ConvexHull

def calculate_convex_hull_area(points):
    """计算凸包面积"""
    if len(points) < 3:
        return 0.0
    
    try:
        hull = ConvexHull(points)
        return hull.volume  # 对于2D点，volume实际上是面积
    except:
        return 0.0

def extend_convex_hull(hull_vertices, extension):
    """向外延伸凸包"""
    if len(hull_vertices) < 3:
        return hull_vertices
    
    extended_vertices = []
    
    for i in range(len(hull_vertices)):
        # 获取当前顶点和相邻顶点
        current = np.array(hull_vertices[i])
        prev = np.array(hull_vertices[(i - 1) % len(hull_vertices)])
        next_vertex = np.array(hull_vertices[(i + 1) % len(hull_vertices)])
        
        # 计算向外的方向
        # 使用相邻边的法向量
        edge1 = current - prev
        edge2 = next_vertex - current
        
        # 归一化
        edge1_norm = edge1 / (np.linalg.norm(edge1) + 1e-8)
        edge2_norm = edge2 / (np.linalg.norm(edge2) + 1e-8)
        
        # 计算向外的法向量（2D情况下的外法向量）
        normal1 = np.array([-edge1_norm[1], edge1_norm[0]])
        normal2 = np.array([-edge2_norm[1], edge2_norm[0]])
        
        # 平均法向量
        avg_normal = (normal1 + normal2) / 2
        avg_normal = avg_normal / (np.linalg.norm(avg_normal) + 1e-8)
        
        # 向外延伸
        extended_vertex = current + extension * avg_normal
        extended_vertices.append(extended_vertex.tolist())
    
    return extended_vertices

def point_in_polygon(point, polygon):
    """检查点是否在多边形内部（使用射线法）"""
    x, y = point
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0]
    for i in range(n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def check_hull_separation(extended_hull, original_points):
    """检查延伸后的凸包是否与原始点分离"""
    if len(extended_hull) < 3:
        return True
    
    # 检查每个原始点是否在延伸凸包内部
    for point in original_points:
        if point_in_polygon(point, extended_hull):
            return False
    
    return True

def check_envelope_separation(screw_positions, extension=0.02):
    """检查螺丝的xy平面外包络线是否不交错（向外延伸extension米）"""
    if len(screw_positions) < 2:
        return True  # 少于2个螺丝，认为成功
    
    # 提取xy坐标
    xy_points = []
    for pos in screw_positions:
        xy_points.append([pos[0], pos[1]])  # 只取x,y坐标
    
    if len(xy_points) < 2:
        return True
    
    # 计算凸包
    try:
        hull = ConvexHull(xy_points)
        
        # 获取凸包的顶点
        hull_vertices = [xy_points[i] for i in hull.vertices]
        
        # 向外延伸extension米
        extended_hull = extend_convex_hull(hull_vertices, extension)
        
        # 检查延伸后的凸包是否重叠
        return check_hull_separation(extended_hull, xy_points)
        
    except Exception as e:
        # 如果凸包计算失败，回退到简单的距离检查
        min_distance = float('inf')
        for i in range(len(xy_points)):
            for j in range(i + 1, len(xy_points)):
                distance = np.linalg.norm(np.array(xy_points[i]) - np.array(xy_points[j]))
                min_distance = min(min_distance, distance)
        
        # 如果最小距离大于2*extension，认为成功
        return min_distance >= 2 * extension

def get_envelope_info(screw_positions):
    """获取外包络线信息"""
    if len(screw_positions) < 2:
        return 0.0, True
    
    # 提取xy坐标
    xy_points = []
    for pos in screw_positions:
        xy_points.append([pos[0], pos[1]])
    
    try:
        hull = ConvexHull(xy_points)
        hull_area = hull.volume  # 对于2D，volume是面积
        is_separated = check_envelope_separation(screw_positions, 0.02)
        return hull_area, is_separated
    except:
        # 回退到简单距离检查
        min_distance = float('inf')
        for i in range(len(xy_points)):
            for j in range(i + 1, len(xy_points)):
                distance = np.linalg.norm(np.array(xy_points[i]) - np.array(xy_points[j]))
                min_distance = min(min_distance, distance)
        
        is_separated = min_distance >= 0.04  # 2 * 0.02
        return min_distance, is_separated 