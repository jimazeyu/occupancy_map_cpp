from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
import sys
sys.path.append('.//DARP')
from multiRobotPathPlanner import *
sys.setrecursionlimit(10000)  # 将默认的递归深度修改为10000

# 外部参数
robot_radius = 4  # 机器人半径
each_block_len = 15  # 划分半径大小
robot_numbers = 0  # 机器人个数
robot_portions = []  # 每个机器人负责检测的区域占比
robot_place = []  # 机器人初始位置

# 原栅格地图参数
occ_occ = 0
occ_unknown = 205
occ_free = 254

occupancy_height = 0
occupancy_width = 0
resolution = 0 #地图分辨率
origin = [] #地图左下角坐标
occupancy_map = 0
occupancy_map_fake = 0 #用于展示的地图

# 网格化地图参数
grid_occ = 1
grid_unknown = 0
grid_free = 2

grid_height = 0
grid_width = 0
grid_map = 0

# color map
color_map = 0

# 一维表示的障碍像素格
linear_points = []

# 是否可视化路径寻找过程
darp_vis = True

# 读取地图
def read_img(name):
    im = Image.open(name)
    return np.array(im)

# 空回调函数
def nothing(x):
    pass

# 点击事件设置机器人初始位置，左键双击选择，右键双击删除
def set_robot_place(event, x, y, flags, param):
    global robot_numbers, robot_portions, robot_place
    # 左键双击
    if event == cv2.EVENT_LBUTTONDBLCLK:
        # 机器人数量+1
        robot_numbers += 1
        # 根据机器人数量平均分配机器人占比
        robot_portions = np.ones(robot_numbers)/robot_numbers
        # 设置机器人初始位置
        grid_x = int(y/each_block_len)
        grid_y = int(x/each_block_len)
        new_pos = int(grid_x*grid_width+grid_y)
        robot_place.append(new_pos)
        print("robot_place:", grid_x, grid_y, x, y)
        # 换算回原地图像素点
        # 检测目标点是否可以作为初始点
        safety_state = check_safety(y,x, robot_radius, occupancy_map)
        print("safety_state:", safety_state)
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        # 清空机器人
        robot_numbers = 0
        # 清空机器人占比
        robot_portions = []
        # 清空机器人初始位置
        robot_place = []

# 生成规划路径
def generate_path(x):
    if(x == 0):
        return
    print("generating path------------")
    # 获取最佳路径
    mpp = MultiRobotPathPlanner(
        grid_height, grid_width, True, robot_place,  robot_portions, linear_points, darp_vis)
    path_list = mpp.get_best_path()
    # 存储路径坐标
    for i, path in enumerate(path_list):
        path_name = "./path/path_"+str(i)+".txt"
        f = open(path_name, "w")
        for x1, y1, x2, y2 in path:
            # 将左下点换算到左上点
            lu_origin_x = origin[0]+occupancy_height*resolution
            x1 = -x1*each_block_len/2*resolution+lu_origin_x
            y1 = y1*each_block_len/2*resolution+origin[1]
            # 保留两位小数
            x1 = round(x1, 2)
            y1 = round(y1, 2)
            f.write(str(x1)+" "+str(y1)+"\n")
        f.close()
    # 保存完成
    print("saving path------------")

# 检测(x,y)在机器人半径周围是否都是空闲
def check_safety(x, y, radius, occupancy_map):
    _x1 = int(x-each_block_len/2)
    _y1 = int(y-each_block_len/2)
    _x2 = int(x+each_block_len/2)
    _y2 = int(y+each_block_len/2)
    _x3 = int(x-each_block_len/2)
    _y3 = int(y+each_block_len/2)
    _x4 = int(x+each_block_len/2)
    _y4 = int(y-each_block_len/2)
    # print(_x1,_x2,_y1,_y2,_x3,_x4,_y3,_y4)
    _map1 = occupancy_map[_x1-radius:_x1+radius, _y1-radius:_y1+radius]
    _map2 = occupancy_map[_x2-radius:_x2+radius, _y2-radius:_y2+radius]
    _map3 = occupancy_map[_x3-radius:_x3+radius, _y3-radius:_y3+radius]
    _map4 = occupancy_map[_x4-radius:_x4+radius, _y4-radius:_y4+radius]
    # print(_map1.shape,_map2.shape,_map3.shape,_map4.shape)
    if np.all(_map1 == occ_free) and np.all(_map2 == occ_free) and np.all(_map3 == occ_free) and np.all(_map4 == occ_free):
        return grid_free
    elif np.all(_map1 == occ_unknown) and np.all(_map2 == occ_unknown) and np.all(_map3 == occ_unknown) and np.all(_map4 == occ_unknown):
        return grid_unknown
    else:
        return grid_occ

# 寻找图像中最大的色块
def find_biggest(color_map, color):
    height = color_map.shape[0]
    width = color_map.shape[1]
    # 测试用图
    tmp_color_map = np.zeros([height, width])
    # 色块是否访问
    vis = np.zeros([height, width])
    # 连通块list
    color_blocks = list()
    # 寻找全部色块

    def dfs(color_map, i, j, lis):
        if(color_map[i][j] != color or vis[i][j]):
            return
        vis[i][j] = 1
        lis.append((i, j))
        # print(lis)
        if(i+1 < height):
            dfs(color_map, i+1, j, lis)
        if(j+1 < width):
            dfs(color_map, i, j+1, lis)
        if(i-1 >= 0):
            dfs(color_map, i-1, j, lis)
        if(j-1 >= 0):
            dfs(color_map, i, j-1, lis)
    # 开始寻找obstacle
    obs_points = []
    for i in range(height):
        for j in range(width):
            if vis[i][j]:
                continue
            if color_map[i][j] == color:
                color_block = []
                dfs(color_map, i, j, color_block)
                color_blocks.append(color_block)
            else:
                obs_points.append((i, j))
    max_len = 0
    max_block = 0
    max_id = 0
    # 除最大联通色块外都当作额外点
    for i, color_block in enumerate(color_blocks):
        if(len(color_block) > max_len):
            max_len = len(color_block)
            max_block = color_block
            max_id = i
    for i, color_block in enumerate(color_blocks):
        if(i == max_id):
            continue
        for point in color_block:
            obs_points.append(point)
    # 测试用
    for x, y in max_block:
        tmp_color_map[x][y] = 2
    plt.imshow(tmp_color_map)
    return obs_points, tmp_color_map

# opencv GUI
def win_name():
    cv2.namedWindow('image')
    cv2.createTrackbar('robot_radius', 'image', 5, 25, nothing)
    cv2.createTrackbar('each_block_len', 'image', 18, 50, nothing)
    # 运行按钮
    switch = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, generate_path)
    # 鼠标点击事件
    cv2.setMouseCallback('image', set_robot_place)
    while True:
        global each_block_len, robot_radius
        # 获取滑条值
        robot_radius = cv2.getTrackbarPos('robot_radius', 'image')
        each_block_len = cv2.getTrackbarPos('each_block_len', 'image')
        # 更新参数
        global grid_height, grid_width, linear_points, grid_map, occupancy_map_fake
        grid_height = int(occupancy_height/each_block_len)
        grid_width = int(occupancy_width/each_block_len)
        # 绘制块地图
        grid_map = np.zeros([grid_height, grid_width])
        for i in range(grid_height):
            for j in range(grid_width):
                _x = i*each_block_len
                _y = j*each_block_len
                grid_map[i][j] = check_safety(
                    _x, _y, robot_radius, occupancy_map)
        # 检测obsatcle
        obs_block, color_map = find_biggest(grid_map, grid_free)
        linear_points = []
        for (x, y) in obs_block:
            linear_points.append(x*grid_width+y)

        # 显示图片
        color_map_show = cv2.resize(
            color_map, (occupancy_width, occupancy_height))
        color_map_show = np.vstack(
            (color_map_show, occupancy_map_fake))  # 上下拼接图片
        # 绘制初始点
        for point in robot_place:
            x = int(point/grid_width)*each_block_len+int(each_block_len/2)
            y = point % grid_width*each_block_len+int(each_block_len/2)
            # circle on gray image
            cv2.circle(color_map_show, (y, x), 10, 0, -1)
            cv2.circle(color_map_show, (y, x+occupancy_height), 10, 0, -1)

        # 改变窗口大小
        cv2.resizeWindow('image', occupancy_width, 2*occupancy_height)
        cv2.imshow('image', color_map_show)
        # waitkey
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()


# main
if __name__ == '__main__':
    # 读取地图
    occupancy_map = read_img("maps//bridge//bridge.pgm")
    # 读取yaml参数
    with open('maps//bridge//bridge.yaml') as f:
        import yaml
        data = yaml.load(f, Loader=yaml.FullLoader)
        resolution = data['resolution']
        origin = data['origin']
    # 初始化展示地图
    occupancy_map_fake = np.copy(occupancy_map)
    # 初始化map参数
    occupancy_height = occupancy_map.shape[0]
    occupancy_width = occupancy_map.shape[1]
    print("occupancy_height:", occupancy_height)
    print("occupancy_width:", occupancy_width)
    # 开启UI
    win_name()