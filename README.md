### Data simulation 
Path: /data_simulation
1. Simulation code of satellite-terrestrial data and task dataset is in **data_simulation**
2. **getSatelliteData.py** is the entry of satellite-ground data simulation.
3. **getTaskData.py** is the entry of task data simulation.

### DataSets 
Path: /data
1. **data.json** is satellite orbit information.
2. **task_{tasks_num}.json** is simulation task data of different orders of magnitude.

### Algorithm implementation
Path: /algo
#### 1. setup:
**sa_env.py:** initializes the environment, performs step according to time, calls reader function in step to read json, after some data processing (incoming actions may not connect,done tasks can not continue to increase done_time, incoming actions lead to each satellite k change).
After the processing is completed, the state is concatenated with the last action recorded by Env to obtain (num_agents *(num_satellites+ 3+3*(num_satellites+4)))),done, etc.). Finally, the reward is calculated.
#### 2. algorithm
DRL-based Algorithm for Task-oriented Satellite Conditional Handover (D-CHO): **mappo.py:** Implement MAPPO, set episode, length, eval, etc., and update the network for eval each time.

