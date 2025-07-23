import json
import numpy as np

from datetime import datetime, timedelta
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
#维护每个时刻下不同位置和卫星之间的连接和可见关系
pos_satellite_map = [set() for _ in range(1+para_dir.num_locations)]# 创建5个位置的卫星状态

#保存当前的卫星状态
satellite_status={}
task_assignments = {}
# 用于存储每个时刻的负载方差
load_variances = []
delay_overhead=[]
utilization_degree=[]
reward=[0 for _ in range(1+para_dir.num_tasks)]

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

    # 优先加入已经开始调度，但是还没有完成的任务
    for task_id,task_assign_info in task_assignments.items():
        #当前任务没有执行完，则重新加入任务列表中
        if tasks_data[task_id-1]['done_time'] < tasks_data[task_id-1]['duration_time']:
            current_tasks.append(tasks_data[task_id-1])

    # 新接入的任务集
    # 如果说存在前面的t-2时刻任务被调度过，但是t-1没有目标卫星，就会被更改time，在添加到现有待完成任务的时候也会因为下标被先加进来
    for task in tasks_data:
        if task['time'] == current_time:
            current_tasks.append(task)



    actual_k_count= [0] * num_satellites
    next_task_assignments={}
    need_handover={}
    for task in current_tasks:
        task_id=task['task_id']
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
            if satellite_status[starlink_id]['actual_k'] <= (satellite_status[starlink_id]['max_K'] * 1 / 4)
        }
        tem_reward=float('-inf')
        selected_satellite=-1
        for sa_id, sa_info in available_satellites.items():
            sa_r = 0.1*0.01 * (tasks_data[task_id-1]['up_data_size'] / satellite_status[sa_id]['transmission_rate'])+ \
                 0.3*0.001*(abs(1-(tasks_data[task_id-1]['duration_time']-tasks_data[task_id-1]['done_time'])/ satellite_status[sa_id]['remain_time'])) + \
                 0.6 * satellite_status[sa_id]['actual_k'] / satellite_status[sa_id]['max_K']
            if task_id in task_assignments and sa_id == task_assignments[task_id]['assigned_satellite']:
                sa_r = 0.5 *0.01* (abs(1 - (tasks_data[task_id - 1]['duration_time'] - tasks_data[task_id - 1]['done_time']) / satellite_status[sa_id]['remain_time'])) + \
                       0.5 * satellite_status[sa_id]['actual_k'] / satellite_status[sa_id]['max_K']
            if sa_r > tem_reward:
                tem_reward=sa_r
                selected_satellite=sa_id

        if selected_satellite == -1 : #or len(next_task_assignments)==150
            next_time = datetime.strptime(current_time, "%Y-%m-%dT%H:%M:%SZ") + timedelta(seconds=para_dir.interval)
            tasks_data[task_id - 1]['time'] = next_time.strftime("%Y-%m-%dT%H:%M:%SZ")
        else:
            reward[task_id - 1]=tem_reward
            # 这个任务是第一次开始执行
            if task_id not in task_assignments:
                satellite_status[selected_satellite]['actual_k'] += 1
                need_handover[task_id] = {
                    'assigned_satellite': selected_satellite,
                    'time': current_time
                }
            elif task_id in task_assignments and selected_satellite != task_assignments[task_id]['assigned_satellite']:
                satellite_status[selected_satellite]['actual_k'] += 1
                satellite_status[task_assignments[task_id]['assigned_satellite']]['actual_k'] -= 1
                need_handover[task_id] = {
                    'assigned_satellite': selected_satellite,
                    'time': current_time
                }
            next_task_assignments[task_id] = {
                'assigned_satellite': selected_satellite,
                'time': current_time
            }
            # 执行完一个任务更新卫星实际连接数量

    task_assignments=next_task_assignments
    # 初始化当前负载计数
    load_counts = [0] * num_satellites  # 初始化全部卫星的负载为0
   # load_counts=[]
    # 计算负载情况
    for starlink_id in satellite_status:
       # load_counts.append(satellite_status[starlink_id]['actual_k'] / satellite_status[starlink_id]['max_K'])
        load_counts[starlink_id - 1] = satellite_status[starlink_id]['actual_k']/satellite_status[starlink_id]['max_K']
    # 计算负载方差
    if load_counts:
        load_variance = np.std(load_counts)
    else:
        load_variance = 0
    load_variances.append(load_variance)


    #计算平均延迟和平均利用率
    current_time_delay=0
    for task_id, task_assign_info in need_handover.items():
        pos_id = task_assign_info['assigned_satellite']
        current_time_delay += tasks_data[task_id - 1]['up_data_size'] / satellite_status[pos_id]['transmission_rate']
   # if len(need_handover) !=0:
   #     current_time_delay /= len(task_assignments)


    current_time_utilize=0
    for task_id,task_assign_info in task_assignments.items():
        sa_id=task_assign_info['assigned_satellite']
        current_time_utilize += abs(1-(tasks_data[task_id-1]['duration_time']-tasks_data[task_id-1]['done_time'])/satellite_status[sa_id]['remain_time'])
    #current_time_utilize /= len(task_assignments)
    delay_overhead.append(current_time_delay)
    utilization_degree.append(current_time_utilize)

    # 输出当前时刻的任务分配结果
    #print(f"Time: {current_time}, Assignments: {task_assignments}")


# 绘制负载方差折线图
utils.plot_load_variances(load_variances)
utils.plot_utilize(utilization_degree)
utils.plot_delay_overhead(delay_overhead)
DP_result={
    'load_variances': load_variances,
    'utilization_degree': utilization_degree,
    'delay_overhead': delay_overhead

}

with open('DP_result_2000.json', 'w') as f:
    json.dump(DP_result, f, indent=4)