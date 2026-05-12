# -*- coding: utf-8 -*-
"""
DDPG for CartPole obstacle avoid with MuJoCo
"""
import math
import mujoco
import mujoco.viewer
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import collections
import random
import copy
import os

max_action = 12.0
max_step = 600
# --- 1. 环境定义 ---
class CartPoleRL:
    def __init__(self, model_path):
        # 确保加载的是带障碍物的模型
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.action_space_high = max_action
        self.obs_direction = 1.0 # 障碍物移动方向

    def reset(self):
        self.count_step = 0
        mujoco.mj_resetData(self.model, self.data)
        # 1. 小车位置
        self.data.qpos[1] = 0.0
        # 2. 障碍物随机位置
        self.data.qpos[0] = -0.5
        # 3. 摆杆垂直向下 (PI)
        self.data.qpos[2] = np.pi 
        
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        # 索引 0:车位, 1:障碍物位, 2:杆角
        x = self.data.qpos[1]
        v = self.data.qvel[1]
        obs_x = self.data.qpos[0]
        obs_v = self.data.qvel[0]
        theta = math.atan2(math.sin(self.data.qpos[2]), math.cos(self.data.qpos[2]))
        omega = self.data.qvel[2]
        
        # 返回 6 维观测：[小车位, 小车速, 杆角, 杆角速, 障碍物位, 相对距离]
        return np.array([
            x, v, theta, omega, obs_v, (x - obs_x)
        ], dtype=np.float32)

    def compute_reward(self, obs, info=None):
        x, v, theta, omega, obs_v, rel_dist = obs
        
        # --- 基础引导奖励 (Shaped Reward) ---
        # 即使没达标，也要引导杆子往上走，小车往中间靠
        r_base = 6 * (np.cos(theta) + 1.0) / 2.0 - 3 * (x**2)
        
        # --- 核心要求 1 & 2：区域目标奖励 ---
        r_target = 0
        is_upright = abs(theta) < 0.1  # 约 5.7 度以内
        is_at_origin = abs(x) < 0.1    # 10cm 以内
        
        if is_upright and is_at_origin:
            r_target += 35.0  # 要求 1：双达标，给最高奖
        elif is_upright:
            r_target += 10.0   # 要求 2：仅角度达标，给中奖
            
        # --- 核心要求 3：惩罚与终点奖励 ---
        r_penalty = 0
        
        r_penalty -= abs(v)
        # 1. 出界惩罚
        if abs(x) >= 1.0:
            r_penalty -= 40.0
            
        if abs(theta) > 120*np.pi/180:
            r_penalty -= 20
            
        # 2. 碰撞惩罚 (通过 info 字典获取步进函数里的碰撞检测)
        if info and info.get("collision"):
            r_penalty -= 100.0  # 给予极大惩罚

        return r_base + r_target + r_penalty 
        
    def step(self, action):
        self.count_step += 1
        # 控制小车
        self.data.ctrl[0] = np.clip(action, -self.action_space_high, self.action_space_high)
        
        # 控制障碍物自动巡逻 (逻辑：触碰边界反转)
        if self.data.qpos[0] < -0.8: self.obs_direction = 1.0
        elif self.data.qpos[0] > 0.8: self.obs_direction = -1.0
        self.data.ctrl[1] = self.obs_direction * 0.5 # 施加恒定推力
        
        mujoco.mj_step(self.model, self.data)
        
        obs = self._get_obs()
        
        over_boundary = abs(self.data.qpos[1]) >= 1.0
        # 检测碰撞
        is_collision = self.data.ncon > 0
        info = {"collision": is_collision,
                "count_step": self.count_step,
                "over_boundary": over_boundary 
                }
        
        # 计算奖励时传入 info
        reward = self.compute_reward(obs, info)
        
        done = abs(self.data.qpos[1]) >= 1.0 or is_collision or self.count_step >= max_step
        return obs, reward, done, info

# --- 2. 网络架构 (State_dim 改为 6) ---
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        self.max_action = max_action

    def forward(self, state):
        a = torch.relu(self.l1(state))
        a = torch.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

    def forward(self, state, action):
        q = torch.relu(self.l1(torch.cat([state, action], 1)))
        q = torch.relu(self.l2(q))
        return self.l3(q)

def soft_update(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        
        
# --- 3. 训练准备 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 使用你刚才修改好的带障碍物的 XML
env = CartPoleRL("../obs_cartpole.xml") 

state_dim = 6  # 修改后的维度
action_dim = 1

actor = Actor(state_dim, action_dim, max_action).to(device)
actor_target = copy.deepcopy(actor).to(device)
optimizer_actor = optim.Adam(actor.parameters(), lr=1e-4)

critic = Critic(state_dim, action_dim).to(device)
critic_target = copy.deepcopy(critic).to(device)
optimizer_critic = optim.Adam(critic.parameters(), lr=1e-3)

replay_buffer = collections.deque(maxlen=100000)

# 超参数
episodes = 2000
batch_size = 64
gamma = 0.99
tau = 0.005
exploration_noise = 5.0 
best_reward = -np.inf

# --- 4. 训练主循环 ---
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    Donereason = "unknown"
    done = False
    for t in range(max_step): # 增加步数，给予躲避时间
    # while(not done):
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = actor(state_tensor).cpu().data.numpy().flatten()
        action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)
        
        next_state, reward, done, info = env.step(action)
        
        if info.get("collision") == True:
            Donereason = "碰撞❌"
        elif info.get("count_step") >= 599 and info.get("collision") == False and abs(next_state[2]) <= 10*np.pi/180:
            Donereason = "完成任务✔"
        elif info.get("over_boundary") == True:
            Donereason = "出界❌"
        else:
            Donereason = "未摆起❌"
        replay_buffer.append((state, action, reward, next_state, done))
        
        state = next_state
        episode_reward += reward
        
        if len(replay_buffer) > batch_size:
            batch = random.sample(replay_buffer, batch_size)
            s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)
            
            s_batch = torch.FloatTensor(np.array(s_batch)).to(device)
            a_batch = torch.FloatTensor(np.array(a_batch)).to(device)
            r_batch = torch.FloatTensor(np.array(r_batch)).reshape(-1, 1).to(device)
            ns_batch = torch.FloatTensor(np.array(ns_batch)).to(device)
            d_batch = torch.FloatTensor(np.array(d_batch)).reshape(-1, 1).to(device)
            
            with torch.no_grad():
                next_action = actor_target(ns_batch)
                target_Q = critic_target(ns_batch, next_action)
                target_Q = r_batch + (gamma * (1 - d_batch) * target_Q)
            
            current_Q = critic(s_batch, a_batch)
            critic_loss = nn.MSELoss()(current_Q, target_Q)
            
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
            
            actor_loss = -critic(s_batch, actor(s_batch)).mean()
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
            
            soft_update(actor, actor_target, tau)
            soft_update(critic, critic_target, tau)

        if done: break

    exploration_noise *= 0.996
    
    # 保存逻辑
    if episode_reward > best_reward and episode > 100:
        best_reward = episode_reward
        torch.save(actor.state_dict(), "../model/best_actor_obs.pth")
        
    print(f"Episode: {episode}, Reward: {episode_reward:.2f}, step_count:{t}, Noise: {exploration_noise:.2f}, Done Reason: {Donereason}")

print("训练完成！已生成支持避障的 Actor 模型。")