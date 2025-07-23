import para_dir
import utils
from datetime import timedelta
from skyfield.api import load, load_file, Topos
import random
import json

# 加载时间尺度
ts = load.timescale()

# 起始时间：2024年7月19日午夜
start_time = ts.utc(para_dir.year, para_dir.month, para_dir.day, para_dir.hour, para_dir.minute, para_dir.second)
# 时间间隔：20分钟
time_interval = timedelta(seconds=para_dir.interval)
# 结束时间：2024年7月19日早上6:00
end_time = start_time + timedelta(seconds=para_dir.obs_time)

# 加载地球数据
data = load_file('de421.bsp')
earth = data['earth']

max_K = para_dir.max_K
actual_k = para_dir.actual_k
# 随机生成 5 个观测点（在北纬 30°-32°，东经 88°-90°范围内）
num_users = para_dir.num_locations
user_points = [
    (random.uniform(para_dir.latitude_st, para_dir.latitude_et), random.uniform(para_dir.longitude_st, para_dir.longitude_et)) for _ in range(num_users)
]

# 将观测点转化为 Topos 对象
user_positions = [earth + Topos(latitude_degrees=lat, longitude_degrees=lon) for lat, lon in user_points]

# 高度角阈值
elevation_threshold = para_dir.elevation_threshold

# 加载卫星数据，筛选符合高度条件的卫星（545 km 到 555 km）
satellites = utils.read_satellites()

# 用于存储所有结果的数据结构
results = []

# 用于存储每个用户对于每个卫星在每个时间点的覆盖状态
user_sat_status = {user_idx: {} for user_idx in range(num_users)}

# 计算覆盖情况
t = start_time
while t < end_time:
    time_entry = {
        "time": t.utc_iso(),
        "positions": []
    }

    for user_idx, user_pos in enumerate(user_positions):
        user_entry = {
            "pos_id": user_idx + 1,
            "latitude": user_points[user_idx][0],
            "longitude": user_points[user_idx][1],
            "satellites": []
        }

        # 遍历所有卫星，判断是否覆盖当前用户
        for sat in satellites:
            eval = user_pos.at(t).observe(earth + sat)  # 卫星相对于用户的观测
            alt, _, _ = eval.apparent().altaz()  # 计算高度角

            if alt.degrees > elevation_threshold:
                altitude = utils.getaltitude(sat)

                # 更新覆盖状态
                if sat.name not in user_sat_status[user_idx]:
                    user_sat_status[user_idx][sat.name] = 0
                user_sat_status[user_idx][sat.name] += 1

                sat_info = {
                    "id": sat.name,
                    "altitude": altitude,
                    "transmission_rate": utils.get_transmission_rate(altitude),
                    "actual_k": actual_k,
                    "max_K": max_K,
                    "remain_time": user_sat_status[user_idx][sat.name]
                }
                user_entry["satellites"].append(sat_info)
            else:
                user_sat_status[user_idx][sat.name] = 0

        # 将用户数据加入到时间条目中
        time_entry["positions"].append(user_entry)

    # 将时间条目添加到结果
    results.append(time_entry)

    # 增加时间
    t += time_interval

# 输出所有结果到 data.json 文件
with open("data.json", "w") as f:
    json.dump(results, f, indent=4)

utils.adjust_remain_time("data.json")
utils.rename_satellite_id("data.json")