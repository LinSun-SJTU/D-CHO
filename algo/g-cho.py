import json
import numpy as np

from datetime import datetime, timedelta,timezone
import para_dir
import utils

# 加载数据
with open('data.json') as f:
    data = json.load(f)

with open('task_2000.json') as f:
    tasks_data = json.load(f)

#tasks_data=tasks_data[:3500]
#卫星数量
num_satellites=data['max_sa_id']
max_K=data['max_K']
#维护每个时刻下不同位置和卫星之间的连接和可见关系
pos_satellite_map = [set() for _ in range(1+para_dir.num_locations)]# 创建5个位置的卫星状态

#保存当前的卫星状态
satellite_status={}
task_assignments = {}
# 用于存储每个时刻的负载方差
load_variances = []
delay_overhead=[]
utilization_degree=[]

#用于记录已完成的任务
done_task={}

# 解析 data.json，初始化卫星的实际连接数量
for entry in data['data']:
    #更新当前任务的done_time
    for task_id,task_assign_info in task_assignments.items():
        if task_assign_info['assigned_satellite'] == None:
            break
        if tasks_data[task_id-1]['done_time'] < tasks_data[task_id-1]['duration_time']:
            tasks_data[task_id - 1]['done_time'] += 1
        if tasks_data[task_id-1]['done_time'] == tasks_data[task_id-1]['duration_time']:
            satellite_status[task_assign_info['assigned_satellite']]['actual_k'] -= 1
            done_task[task_id]=task_assign_info
            tasks_data[task_id - 1]['done_time'] += 1

    current_time = entry['time']
    positions = entry['positions']

    for position in positions:
        pos_id = position['pos_id']
        current_pos_satellites=set()
        satellites = position['satellites']

        # 为每个卫星初始化状态并记录可见性
        for satellite in satellites:
            starlink_id = satellite['id']
            #由于是set，所以直接添加即可，如果上个时刻还在的卫星，会被去重
            pos_satellite_map[pos_id].add(starlink_id)

            #如果是新的卫星，添加到当前的卫星状态映射中
            if starlink_id not in satellite_status:
                satellite_status[starlink_id] = {
                    'remain_time': satellite['remain_time'],
                    'transmission_rate': satellite['transmission_rate'],
                    'max_K': satellite['max_K'],
                    'actual_k': 0
                }
            else:
                # 更新当前时间和剩余时间
                satellite_status[starlink_id]['remain_time'] = satellite['remain_time']

            current_pos_satellites.add(starlink_id)

        remove_sa=[]
        for sa in pos_satellite_map[pos_id]:
            if sa not in current_pos_satellites:
                remove_sa.append(sa)
        for sa in remove_sa:
            pos_satellite_map[pos_id].remove(sa)

    current_tasks=[]#当前待执行的任务分为下面两部分

    current_time_utilize = 0
    # 卫星不可见的任务集
    for task_id,task_assign_info in task_assignments.items():
        #获得当前任务的位置id
        pos_id=tasks_data[task_id-1]['pos_id']
        #当前任务没有执行完，判断塔的卫星是否还存活，没有存活的话就重新进入卫星执行列表
        if tasks_data[task_id-1]['done_time'] < tasks_data[task_id-1]['duration_time']:
            selected_satellite=task_assign_info['assigned_satellite']
            if selected_satellite not in pos_satellite_map[pos_id]:
                current_tasks.append(tasks_data[task_id-1])
            else:
                current_time_utilize+= abs(1- (tasks_data[task_id-1]['duration_time']- tasks_data[task_id-1]['done_time'] )/satellite_status[selected_satellite]['remain_time'])

    # 新接入的任务集
    for task in tasks_data:
        if task['time'] == current_time:
            current_tasks.append(task)

    # 贪心算法选择卫星
    #actual_k_count= [0] * num_satellites
    current_time_tasks_handover={}
    for task in current_tasks:

        pos_id = task['pos_id']
        # 找到可以连接的卫星
        #在当前pos_satellite_map['pos_id']中的卫星，一定是对pos可见的
        available_satellites = {
            starlink_id: {
                'actual_k': satellite_status[starlink_id]['actual_k'],
                'remain_time': satellite_status[starlink_id]['remain_time'],
                'max_K': satellite_status[starlink_id]['max_K']
            }
            for starlink_id in pos_satellite_map[pos_id]
            if satellite_status[starlink_id]['actual_k'] <= (satellite_status[starlink_id]['max_K'] * 1/4)
        }

        if available_satellites:
            # 选择负载最少的卫星
            selected_satellite = min(
                available_satellites.items(),
                key=lambda item: (item[1]['actual_k']/item[1]['max_K'])
            )[0]  # 选择实际连接数量少且剩余时间最长的卫星

            # 记录此次新增卫星的实际连接数量
           # actual_k_count[selected_satellite-1] += 1

            # 记录任务分配结果
            task_assignments[task['task_id']] = {
                'assigned_satellite': selected_satellite,
                'time': current_time
            }
            #添加当此次的移交列表中，计算任务平均延迟
            current_time_tasks_handover[task['task_id']] = {
                'assigned_satellite': selected_satellite,
                'time': current_time
            }
            # 执行完一个任务更新卫星实际连接数量
            satellite_status[selected_satellite]['actual_k'] += 1
        else:
            next_time = datetime.strptime(current_time, "%Y-%m-%dT%H:%M:%SZ")+timedelta(seconds=para_dir.interval)
            tasks_data[task['task_id']-1]['time'] = next_time.strftime("%Y-%m-%dT%H:%M:%SZ")

    # 初始化当前负载计数
    load_counts = [0] * num_satellites  # 初始化全部卫星的负载为0
    #load_counts=[]
    # 计算负载情况
    for starlink_id in satellite_status:
        #load_counts.append(satellite_status[starlink_id]['actual_k']/satellite_status[starlink_id]['max_K'])
        load_counts[starlink_id - 1] = satellite_status[starlink_id]['actual_k']/satellite_status[starlink_id]['max_K']
    # 计算负载方差
    if load_counts:
        load_variance = np.std(load_counts)
    else:
        load_variance = 0
    load_variances.append(load_variance)


# 只要任务开始执行，只会在卫星飞走的时候，才执行切换，所以当前的切换的列表就一定需要计算切换成本
    #计算平均延迟和平均利用率
    current_time_tasks_handover_num=len(current_time_tasks_handover)
    current_time_delay=0

    for task_id,task_assign_info in current_time_tasks_handover.items():
        pos_id=task_assign_info['assigned_satellite']
        current_time_delay += tasks_data[task_id-1]['up_data_size']/satellite_status[pos_id]['transmission_rate']
        current_time_utilize += abs(1-(tasks_data[task_id-1]['duration_time']-tasks_data[task_id-1]['done_time'])/satellite_status[pos_id]['remain_time'])
#    if current_time_tasks_handover_num !=0:
#       current_time_delay /= current_time_tasks_handover_num
    delay_overhead.append(current_time_delay)
    utilization_degree.append(current_time_utilize)

    # 输出当前时刻的任务分配结果
    #print(f"Time: {current_time}, Assignments: {task_assignments}")


# 绘制负载方差折线图
utils.plot_load_variances(load_variances)
utils.plot_utilize(utilization_degree)
utils.plot_delay_overhead(delay_overhead)

GSH_result={
    'load_variances': load_variances,
    'utilization_degree': utilization_degree,
    'delay_overhead': delay_overhead

}

with open('GSH_result_2000.json', 'w') as f:
    json.dump(GSH_result, f, indent=4)