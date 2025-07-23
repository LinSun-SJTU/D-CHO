import numpy as np
import torch
import torch.nn as nn

import reader
from sa_env import CustomEnv

class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        # 网络结构：输入观测 -> 输出动作概率分布
        self.fc = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),  # 激活函数
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)  # 确保输出为合法概率分布
        )

    def forward(self, obs):
        return self.fc(obs)

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(Critic, self).__init__()
        # 网络结构：输入全局状态 -> 输出标量价值
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.fc(state).squeeze(-1)  # 输出形状 [batch_size]

class MAPPO:
    def __init__(self, env, device,hidden_dim, obs_dim,action_dim,state_dim,gamma, clip, lr,episode_length):
        # 环境参数
        self.env = env
        self.num_agents = env.num_agents
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.obs_dim=obs_dim
        self.device=device
        self.episode_length=episode_length

        # 算法参数
        self.gamma = gamma
        self.clip = clip

        # 初始化网络
        self.actors = [
            Actor(self.obs_dim, self.action_dim,hidden_dim).to(device)
            for _ in range(self.num_agents)
        ]
        self.critics = [
            Critic(self.state_dim, hidden_dim).to(device)
            for _ in range(self.num_agents)
        ]

        # 优化器
        self.actor_optimizers = [torch.optim.Adam(actor.parameters(), lr=lr) for actor in self.actors]
        self.critic_optimizers = [torch.optim.Adam(critic.parameters(), lr=lr) for critic in self.critics]

        # 经验缓存（存储整个episode的数据）
        self.buffer = []

    def act(self, obs):
        """为每个智能体生成动作和旧概率"""
        actions = []
        one_hot_actions=[]
        old_probs=[]
        select_probs = []

        for agent_id in range(self.num_agents):
            # 若模型在GPU上运行，将输入数据显式转移到GPU
            obs_tensor = obs[agent_id].to(self.device)
           # obs_tensor = torch.tensor(obs[agent_id], dtype=torch.float32)
            action_probs = self.actors[agent_id](obs_tensor)
            # 该函数根据输入的概率分布action_probs 进行‌按概率权重的随机采样‌：
            #若 action_probs = [0.8, 0.2]，则动作0被选中的概率为80%，动作1为20%；
            # 若 action_probs = [0.5, 0.5]，则动作0和1的选中概率均等（均匀随机）。
            action = torch.multinomial(action_probs, 1).item()

            # 记录选择动作的概率
            select_prob = action_probs[action].detach().item()
            old_probs.append(action_probs)

            actions.append(action)
            one_hot_actions.append(reader.get_onehot(action, self.action_dim))
            select_probs.append(select_prob)
        old_probs=torch.stack(old_probs, dim=0)
        #one_hot_actions:用于适配最初MADDPG算法环境采样时的输入，为num_agent*action_dim的二维数组，而不是具体某个选中的卫星编号
        return actions,np.array(one_hot_actions), old_probs, select_probs

    def store_transition(self, transition):
        """存储单步经验"""
        self.buffer.append(transition)

    def compute_returns_advantages(self, rewards, dones, values):
        """计算蒙特卡洛回报和优势"""
        returns = []
        advantages = []
        R = 0

        # 反向计算
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns.insert(0, R)
            advantage = R - values[t]
            advantages.insert(0, advantage)

        return returns, advantages

    def update(self):
        """更新所有智能体的策略"""
        # 将buffer数据转换为张量
       # for t in self.buffer:
         #   states = torch.stack(t['state'], dim=0).to(self.device)
        states = torch.stack([t['state'] for t in self.buffer]).to(self.device)
        flat_states = states.view(self.episode_length, -1)
        probs_agent = torch.stack([t['old_probs'] for t in self.buffer]).to(self.device)
        flat_probs_agent=probs_agent.view(self.episode_length, -1)
        input_state = torch.cat((flat_states, flat_probs_agent), dim=1)
        input_state = input_state.detach()
        rewards = np.array([t['rewards'].cpu().numpy() for t in self.buffer])  # [T, n_agents]
        dones = np.array([t['dones'].cpu().numpy() for t in self.buffer])
        # 对每个智能体独立更新
        for agent_id in range(self.num_agents):
            #print('当前更新到 agent:',agent_id)
            # 提取该智能体的数据
            obs_agent = torch.stack([t['obs'][agent_id] for t in self.buffer]).to(self.device)
            actions_agent = torch.stack([
                torch.tensor(t['actions'][agent_id], dtype=torch.long)  # 转换为 LongTensor
                for t in self.buffer
            ]).to(self.device)

            old_probs_agent = torch.stack([
                torch.tensor(t['select_probs'][agent_id], dtype=torch.float)  # 转换为 LongTensor
                for t in self.buffer
            ]).to(self.device)
            #一共是episode_length个reward，取出来当前agent_id的全部reward
            rewards_agent = rewards[:, agent_id]

            # --- Critic更新 ---
            # 计算价值预测

            values_pred = self.critics[agent_id](input_state).squeeze()
            values = values_pred.detach().cpu().numpy()
            returns, _ = self.compute_returns_advantages(rewards_agent, dones[:, agent_id], values)
            returns = torch.FloatTensor(returns).to(self.device)

            # 计算Critic损失
            critic_loss = nn.MSELoss()(values_pred, returns)
            self.critic_optimizers[agent_id].zero_grad()
            critic_loss.backward()

            self.critic_optimizers[agent_id].step()

            # --- Actor更新 ---
            # 计算新概率
            new_probs = self.actors[agent_id](obs_agent).gather(1, actions_agent.unsqueeze(1)).squeeze()
            ratio = new_probs / old_probs_agent

            # 计算优势函数
            _, advantages = self.compute_returns_advantages(rewards_agent, dones[:, agent_id], values)
            advantages = torch.FloatTensor(advantages).to(self.device)

            # PPO裁剪损失
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            self.actor_optimizers[agent_id].zero_grad()
            actor_loss.backward()
            self.actor_optimizers[agent_id].step()

        # 清空缓存
        self.buffer = []


def make_env(num_tasks):
    algo='mappo'
    num_agents = 150
    num_satellites = 29
    env = CustomEnv(num_agents, num_satellites,num_tasks,algo)
    return env

# 使用示例
if __name__ == "__main__":
    # 假设已实例化自定义环境
    num_episodes = 500
    episode_length = 60  # 每条探索序列的最大长度
    buffer_size = 1000
    hidden_dim = 256
    lr = 1e-2
    gamma = 0.95
    clip = 0.2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    update_interval = 240
    minimal_size = 120

    #mappo算法中obs是actor的输入，state是critic的输入
    obs_dim = 131
    action_dim = 3
   # num_tasks=int(input("请输入任务数量:\n"))
    env = make_env(3500)  # 用户需自行实现环境
    state_dim = (obs_dim+action_dim)*env.num_agents
    mappo = MAPPO(env, device,hidden_dim, obs_dim,action_dim,state_dim,gamma, clip, lr,episode_length)

    for episode in range(num_episodes):
        state = env.reset()
       # episode_rewards = np.zeros(env.num_agents)
        for e_i in range(episode_length):
            # 获取所有智能体的观测
            # 选择动作并执行
            actions, one_hot_actions,old_probs, select_probs = mappo.act(state)
            next_state, rewards, dones, truncated, info = env.step(one_hot_actions, time=e_i)
            # 存储经验
            transition = {
                'state': state,
                'obs': state,
                'actions': actions,
                'rewards': rewards,
                'dones': dones,
                'old_probs': old_probs,
                'select_probs': select_probs
            }
            mappo.store_transition(transition)
            state = next_state
           # episode_rewards += np.array(rewards)
        # 更新策略
        mappo.update()
        print(f"Episode {episode + 1} finish!")
