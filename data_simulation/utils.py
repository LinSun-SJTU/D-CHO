import numpy as np
import json
from matplotlib import pyplot as plt


from skyfield.api import load
from skyfield.iokit import parse_tle_file

import para_dir


#模拟生成卫星对某点的传输速率
def get_transmission_rate(altitude):
    # 定义高度范围和传输速率范围
    min_altitude = para_dir.min_altitude
    max_altitude = para_dir.max_altitude
    #starlink的上传数据速率范围为 50～200 Mbps
    min_rate = para_dir.min_rate
    max_rate = para_dir.max_rate

    # 检查是否在定义的高度范围内
    if altitude < para_dir.min_altitude:
        return min_rate
    elif altitude > max_altitude:
        return max_rate

    # 根据线性插值计算传输速率
    # (y - min_rate) / (altitude - min_altitude) = (max_rate - min_rate) / (max_altitude - min_altitude)
    #rate = min_rate + ((altitude - min_altitude) / (max_altitude - min_altitude)) * (max_rate - min_rate)
    rate = max_rate - ((altitude - min_altitude) / (max_altitude - min_altitude)) * (max_rate - min_rate)

    return rate

#直接解析卫星数据得到海拔
def getaltitude(sat):
    lap = velocitytolaps(sat)
    return lapstoaltitude(lap)

#通过速率得到海拔高度
def lapstoaltitude(laps):
    period = 86400.0 / laps

    coeff = 21613.546
    orbit_radius = coeff * np.power(period, 2/3)
    earth_radius = 6371.393e3
    altitude = (orbit_radius - earth_radius) / 1e3
    return altitude

#解析当前卫星数据，获得速率参数
def velocitytolaps(sat):
    laps_per_day = sat.model.no_kozai * 60 * 24 / (2*np.pi)
    return laps_per_day


def read_satellites(file_name="starlink.tle"):
    ts = load.timescale()
    with load.open("starlink.tle") as f:
        satellites = list(parse_tle_file(f, ts))
    #获得数组前1584个元素
    satellites = satellites[: 1584]

    satellites_v1 = []
    for satellite in satellites:
        lap = velocitytolaps(satellite)
        altitude = lapstoaltitude(lap)
        if para_dir.min_altitude <= altitude <= para_dir.max_altitude:
            satellites_v1.append(satellite)
    print("The Starlink constellation consists of {} satellites".format(len(satellites_v1)))
    return satellites_v1


def test():
    ts = load.timescale()
    with load.open("starlink.tle") as f:
        satellites = list(parse_tle_file(f, ts))
    satellite = satellites[1582]
    lap = velocitytolaps(satellite)
    altitude = lapstoaltitude(lap)
    print(f"the altitude is {altitude}")

    altitudes = []
    right_ascensions = []
    inclination = []
    for i in range(1584):
        lap = velocitytolaps(satellites[i])
        altitude = lapstoaltitude(lap)
        altitudes.append(altitude)
        right_ascensions.append(satellites[i].model.nodeo*180/np.pi)
        inclination.append(satellites[i].model.inclo*180/np.pi)

    x = np.arange(1, len(altitudes)+1)

    fig, ax = plt.subplots(2, 2)
    ax[0, 0].scatter(x, altitudes)
    ax[0, 1].scatter(x, right_ascensions)
    ax[1, 0].scatter(x, inclination)
    plt.show()


def adjust_remain_time(file_path):
    # 读取已有的 data.json 文件
    with open(file_path, 'r') as f:
        data = json.load(f)

    # 逐步查看每个时间戳，为相同用户的相同卫星记录连续覆盖的状态
    for user_index in range(len(data[0]['positions'])):
        # 第一个字典是保存同一个用户对同一个卫星的连接关系
        sat_remain_map = {}
        #第二个字典是重新计数，同时是逆序遍历json文件，就可以调整原来的累计连接时间改为剩余连接时间
        sat_remain_counter={}

        # 倒序查看不同时间戳的用户和卫星数据
        for entry in reversed(data):
            user_data = entry['positions'][user_index]

            for sat in user_data['satellites']:
                sat_id = sat['id']
                current_remain_time = sat['remain_time']

                # 如果我们没记录这个卫星，或者current_remain_time是1，那么就要创建
                if sat_id not in sat_remain_map:
                    sat_remain_map[sat_id] = current_remain_time
                    sat_remain_counter[sat_id]=1

                # 记录在最新索引的时间
                sat['remain_time'] = sat_remain_counter[sat_id]

                # 向下计数，达到1时重新开始计算
                if sat_remain_map[sat_id] > 1:
                    sat_remain_map[sat_id] -= 1
                    sat_remain_counter[sat_id] += 1
                else:
                    del sat_remain_map[sat_id]
                    del sat_remain_counter[sat_id]

    # 写入修改后的数据回到 data.json 文件
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def rename_satellite_id(file_path):

    # 读取JSON文件
    with open(file_path, 'r') as file:
        data = json.load(file)

    # 用于存储已分配的卫星ID及对应的new_id
    satellite_ids = {}
    next_id = 1

    # 遍历所有的time节点
    for entry in data:
        # 遍历所有position节点
        for position in entry['positions']:
            # 遍历该定位点的每个卫星
            for satellite in position['satellites']:
                # 将原来的id重命名为starlink_id
                satellite['starlink_id'] = satellite.pop('id')

                starlink_id = satellite['starlink_id']

                # 检查该starlink_id是否已有对应的new_id
                if starlink_id not in satellite_ids:
                    satellite_ids[starlink_id] = next_id
                    next_id += 1
                # 分配或查找new_id并重命名为id
                satellite['id'] = satellite_ids[starlink_id]

    # 计算最大的ID
    max_sa_id = next_id - 1

    # 创建新的数据结构，并将原有数据存放在data属性内
    new_data_structure = {
        "max_sa_id": max_sa_id,
        "data": data
    }

    # 将修改后的数据保存回原JSON文件
    with open(file_path, 'w') as file:
        json.dump(new_data_structure, file, indent=4)

    print("Updated satellite IDs, added max_sa_id, and saved back to data.json.")


