import numpy as np
from datetime import datetime, timedelta,timezone
import random
import json

import para_dir


def generate_fixed_interval_tasks(start_time, obs_time, interval_seconds, num_tasks):
    # 计算每个时间点需要生成的任务数
    total_intervals = 1+int(obs_time / interval_seconds)
    tasks_per_interval = num_tasks // total_intervals
    # 根据步骤生成时间戳列表
    times = []

    for i in range(total_intervals):
        time_point = start_time + timedelta(seconds=i * interval_seconds)
        # 件一个时间点，为每个时间点加上标准任务数量
        for _ in range(tasks_per_interval):
            times.append(time_point)
    dummy=num_tasks-len(times)
    for i in range(dummy):
        times.append(start_time + timedelta(seconds=obs_time))
    return times

def generate_poisson_times(rate, start_time, end_time, interval, num_tasks):
    times = []
    current_time = start_time

    while current_time <= end_time:
        num_generated = np.random.poisson(rate)

        if len(times) < num_tasks:  # Check how many more tasks we need
            task_times = [current_time] * min(num_generated, num_tasks - len(times))
            times.extend(task_times)

        current_time += interval

    # If we have not generated enough tasks, fill them with random timestamps
    if len(times) < num_tasks:
        additional_times = np.random.choice(times, num_tasks - len(times), replace=True)
        times.extend(additional_times)

    return times[:num_tasks]

def calculate_data_volume():
    rate = random.uniform(para_dir.min_rate, para_dir.max_rate)  # Mbps
    time = random.uniform(0.05, 0.125)  # seconds

    data_volume = rate * time   # MB---->KB
    return int(data_volume)  # Convert to integer

def utc_time_invert(dt):

    # 为 datetime 对象添加 UTC 时区信息
    dt_utc = dt.replace(tzinfo=timezone.utc)

    # 格式化为所需的格式字符串，显示到秒
    formatted_time_str = dt_utc.isoformat(timespec='seconds').replace('+00:00', 'Z')
    return formatted_time_str

def generate_tasks(num_tasks, num_locations):
    # 时间范围
    start_time = datetime(para_dir.year, para_dir.month, para_dir.day, para_dir.hour, para_dir.minute, para_dir.second)
    interval = timedelta(seconds=para_dir.interval)

    end_time = start_time + timedelta(seconds=para_dir.obs_time)

    # 计算需要的泊松参数
    time_rate = num_tasks // ((end_time - start_time).seconds // 20 + 1)
    location_rate = num_tasks / num_locations

    # 获得任务的时间
   # times = generate_poisson_times(time_rate, start_time, end_time, interval, num_tasks)
    times=generate_fixed_interval_tasks(start_time, para_dir.obs_time, para_dir.interval, num_tasks)

    # 生成任务
    tasks = []

    for i in range(num_tasks):
        task = {
            'task_id': i+1,
            'time': utc_time_invert(times[0]),
            'pos_id': np.random.poisson(location_rate) % num_locations + 1,
            'up_data_size': calculate_data_volume(),  # 数据量可按需调整
            'duration_time': random.randint(1,4 ),  # 数据量可按需调整
            'done_time': 0
        }
        tasks.append(task)

    return tasks


# 生成并打印任务
tasks = generate_tasks(num_tasks=para_dir.num_tasks, num_locations=para_dir.num_locations)
with open('task.json', 'w') as f:
    json.dump(tasks, f, indent=4)