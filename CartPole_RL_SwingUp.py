# -*- coding: utf-8 -*-
"""
DDPG for CartPole Swing-up with MuJoCo
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

# --- 1. 环境定义 ---
class CartPoleRL:
    def __init__(self, model_path):
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.action_space_high = 300.0  

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)
        # 初始化为垂直向下，并给一点随机扰动
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = np.pi + np.random.uniform(-0.1, 0.1)
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs()

    def _get_obs(self):
        # 观察值：[车位, 车速, 角度, 角速度]
        # 注意：这里你可以考虑使用 sin(theta), cos(theta) 代替 angle 以获得更好的收敛性
        pole_angle = math.atan2(math.sin(self.data.qpos[1]), math.cos(self.data.qpos[1]))
        return np.array([
            self.data.qpos[0], self.data.qvel[0],
            pole_angle, self.data.qvel[1]
        ], dtype=np.float32)

    def compute_reward(self, obs):
        x, x_dot, theta, theta_dot = obs
        # 奖励设计：角度接近 0 (向上) 奖励大，尽量保持在中心，减少过大的控制力
        # 使用 cos(theta) 是起摆任务的常用技巧
        reward = np.cos(theta) - 0.01 * (x**2) - 0.001 * (theta_dot**2)
        return reward

    def step(self, action):
        self.data.ctrl[0] = np.clip(action, -self.action_space_high, self.action_space_high)
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self.compute_reward(obs)
        # 车跑出界（例如 ±1.5m）则结束
        done = abs(self.data.qpos[0]) >= 1.0
        return obs, reward, done, {}

# --- 2. 网络架构 ---
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
env = CartPoleRL("cartpole.xml") # 请确保路径正确

state_dim = 4
action_dim = 1
max_action = 300.0

# 实例化网络
actor = Actor(state_dim, action_dim, max_action).to(device)
actor_target = copy.deepcopy(actor).to(device)
optimizer_actor = optim.Adam(actor.parameters(), lr=1e-4)

critic = Critic(state_dim, action_dim).to(device)
critic_target = copy.deepcopy(critic).to(device)
optimizer_critic = optim.Adam(critic.parameters(), lr=1e-3)

replay_buffer = collections.deque(maxlen=100000)

# 超参数
episodes = 3000
batch_size = 64
gamma = 0.99
tau = 0.005
exploration_noise = 20.0 # 初始探索噪声
best_reward = -np.inf

# --- 4. 训练主循环 ---
for episode in range(episodes):
    state = env.reset()
    episode_reward = 0
    
    for t in range(500):
        # 1. 策略选择 + 探索噪声
        state_tensor = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = actor(state_tensor).cpu().data.numpy().flatten()
        action = (action + np.random.normal(0, exploration_noise, size=action_dim)).clip(-max_action, max_action)
        
        # 2. 与环境交互
        next_state, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, next_state, done))
        
        state = next_state
        episode_reward += reward
        
        # 3. 核心训练逻辑
        if len(replay_buffer) > batch_size:
            # 采样
            batch = random.sample(replay_buffer, batch_size)
            s_batch, a_batch, r_batch, ns_batch, d_batch = zip(*batch)
            
            s_batch = torch.FloatTensor(np.array(s_batch)).to(device)
            a_batch = torch.FloatTensor(np.array(a_batch)).to(device)
            r_batch = torch.FloatTensor(np.array(r_batch)).reshape(-1, 1).to(device)
            ns_batch = torch.FloatTensor(np.array(ns_batch)).to(device)
            d_batch = torch.FloatTensor(np.array(d_batch)).reshape(-1, 1).to(device)
            
            # --- 更新 Critic ---
            with torch.no_grad():
                # Target Q = r + gamma * (1-done) * Critic_Target(next_state, Actor_Target(next_state))
                next_action = actor_target(ns_batch)
                target_Q = critic_target(ns_batch, next_action)
                target_Q = r_batch + (gamma * (1 - d_batch) * target_Q)
            
            current_Q = critic(s_batch, a_batch)
            critic_loss = nn.MSELoss()(current_Q, target_Q)
            
            optimizer_critic.zero_grad()
            critic_loss.backward()
            optimizer_critic.step()
            
            # --- 更新 Actor ---
            # 目标是最大化 Critic 的输出，即最小化 -Critic
            actor_loss = -critic(s_batch, actor(s_batch)).mean()
            
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()
            
            # --- 软更新目标网络 ---
            soft_update(actor, actor_target, tau)
            soft_update(critic, critic_target, tau)

        if done: break

    # 噪声衰减，后期需要更精确的控制
    exploration_noise *= 0.995 
    
    # 保存逻辑
    if episode_reward > best_reward:
        best_reward = episode_reward
        torch.save(actor.state_dict(), "best_actor.pth")
        
    if episode % 20 == 0:
        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Noise: {exploration_noise:.2f}")

print("训练完成！最优模型已保存为 best_actor.pth")