import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

from torch.distributions import Categorical

GAMMA = 0.95
# 奖励折扣因子
LR = 0.01
# 学习率
EPISODE = 3000
# 生成多少个episode
STEP = 3000
# 一个episode里面最多多少步
TEST = 10
# 每100步episode后进行测试，测试多少个

class PGNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PGNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        action_scores = self.fc2(x)
        return F.softmax(action_scores, dim=1)
    # PGNetwork的作用是输入某一时刻的state向量，输出是各个action被采纳的概率
    # 和policy gradient中的Policy一样


class Actor(object):

    def __init__(self, env):
        # 初始化

        self.state_dim = env.observation_space.shape[0]
        # 表示某一时刻状态是几个维度组成的
        # 在推杆小车问题中，这一数值为4

        self.action_dim = env.action_space.n
        # 表示某一时刻动作空间的维度（可以有几个不同的动作）
        # 在推杆小车问题中，这一数值为2

        self.network = PGNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim)
        # 输入S输出各个动作被采取的概率

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)

    def choose_action(self, observation):
        # 选择动作，这个动作不是根据Q值来选择，而是使用softmax生成的概率来选
        #  在policy gradient和A2C中，不需要epsilon-greedy，因为概率本身就具有随机性
        observation = torch.from_numpy(observation).float().unsqueeze(0)
        # print(state.shape)
        # torch.size([1,4])
        # 通过unsqueeze操作变成[1,4]维的向量

        probs = self.network(observation)
        # Policy的返回结果，在状态x下各个action被执行的概率

        m = Categorical(probs)
        # 生成分布

        action = m.sample()
        # 从分布中采样（根据各个action的概率）

        # print(m.log_prob(action))
        # m.log_prob(action)相当于probs.log()[0][action.item()].unsqueeze(0)
        # 换句话说，就是选出来的这个action的概率，再加上log运算

        return action.item()
        # 返回一个元素值

        '''
        所以每一次select_action做的事情是，选择一个合理的action,返回这个action;
        '''

    def learn(self, state, action, td_error):
        observation = torch.from_numpy(state).float().unsqueeze(0)

        softmax_input = self.network(observation)
        # 各个action被采取的概率

        action = torch.LongTensor([action])
        # neg_log_prob = F.cross_entropy(input=softmax_input, target=action)
        '''
        注：这里我原来用的是cross_entropy，但用这个loss_function是不对的
        因为cross_entropy中会再进行一次softmax
        而实际上我们只需要计算logP(a|s)即可
        所以改成：
        '''
        l = torch.nn.NLLLoss()
        log_softmax_input = torch.log(softmax_input)
        neg_log_prob = l(log_softmax_input, action)

        # 反向传播（梯度上升）
        # 这里需要最大化当前策略的价值
        # 因此需要最大化neg_log_prob * td_error,即最小化-neg_log_prob * td_error
        loss_a = -neg_log_prob * td_error

        self.optimizer.zero_grad()
        loss_a.backward()
        self.optimizer.step()
        # pytorch 老三样


class QNetwork(nn.Module):
    """
    critic 主干网络，
    输入为state，
    输出为状态值
    """
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 20)
        self.fc2 = nn.Linear(20, 1)
        # 这个地方和之前略有区别，输出不是动作维度，而是一维
        # 因为我们这里需要计算的是V(s),而在DQN中，是Q(s,a),所以那里是两维，这里是一维

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return out


class Critic(object):
    # 通过采样数据，学习V(S)
    def __init__(self, env):
        self.state_dim = env.observation_space.shape[0]
        # 表示某一时刻状态是几个维度组成的
        # 在推杆小车问题中，这一数值为4

        self.action_dim = env.action_space.n
        # 表示某一时刻动作空间的维度（可以有几个不同的动作）
        # 在推杆小车问题中，这一数值为2

        self.network = QNetwork(
            state_dim=self.state_dim,
            action_dim=self.action_dim)
        # 输入S，输出V(S)

        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def train_Q_network(self, state, reward, next_state):
        # 类似于DQN的5.4，不过这里没有用fixed network，experience relay的机制

        s, s_ = torch.FloatTensor(state), torch.FloatTensor(next_state)
        # 当前状态，执行了action之后的状态

        v = self.network(s)  # v(s)
        v_ = self.network(s_)  # v(s')

        # 反向传播
        loss_q = self.loss_func(reward + GAMMA * v_, v)
        # TD
        ##r+γV(S') 和V(S) 之间的差距

        self.optimizer.zero_grad()
        loss_q.backward()
        self.optimizer.step()
        # pytorch老三样

        with torch.no_grad():
            td_error = reward + GAMMA * v_ - v
        # 表示不把相应的梯度传到actor中（actor和critic是独立训练的）

        return td_error


def main():
    env = gym.make('CartPole-v0')
    # 创建一个推车杆的gym环境

    actor = Actor(env)
    critic = Critic(env)

    for episode in range(EPISODE):
        state = env.reset()
        # state表示初始化这一个episode的环境

        for step in range(STEP):
            action = actor.choose_action(state)
            # 根据actor选择action

            next_state, reward, done, _ = env.step(action)
            # 四个返回的内容是state,reward,done(是否重置环境),info

            td_error = critic.train_Q_network(
                state,
                reward,
                next_state)
            # gradient = grad[r + gamma * V(s_) - V(s)]
            # 先根据采样的action，当前状态，后续状态，训练critic，以获得更准确的V（s）值

            actor.learn(state, action, td_error)
            # true_gradient = grad[logPi(a|s) * td_error]
            # 然后根据前面学到的V（s）值，训练actor，以更好地采样动作

            state = next_state
            if done:
                break

        # 每100步测试效果
        if episode % 100 == 0:
            total_reward = 0
            for i in range(TEST):
                state = env.reset()
                for j in range(STEP):
                    # env.render()
                    # 渲染环境，如果你是在服务器上跑的，只想出结果，不想看动态推杆过程的话，可以注释掉

                    action = actor.choose_action(state)
                    # 采样了一个action

                    state, reward, done, _ = env.step(action)
                    # 四个返回的内容是state,reward,done(是否重置环境),info
                    total_reward += reward
                    if done:
                        break
            ave_reward = total_reward / TEST
            print('episode: ', episode, 'Evaluation Average Reward:', ave_reward)


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    print('Total time is ', time_end - time_start, 's')

