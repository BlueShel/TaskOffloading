import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
torch.manual_seed(1)

class DoubleDQN(nn.Module):
    def __init__(self, n_actions, n_features, learning_rate=0.005, reward_decay=0.9,
                 e_greedy=0.9, replace_target_iter=200, memory_size=3000,
                 batch_size=32, e_greedy_increment=None, double_q=True):
        super(DoubleDQN, self).__init__()
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max
        self.double_q = double_q

        # Total learning step
        self.learn_step_counter = 0

        # Initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, n_features * 2 + 2))

        # Create two networks (target and evaluate)
        self.eval_net = nn.Sequential(
            nn.Linear(n_features, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, n_actions)
        )

        self.target_net = nn.Sequential(
            nn.Linear(n_features, 20),
            nn.ReLU(),
            nn.Linear(20, 20),
            nn.ReLU(),
            nn.Linear(20, n_actions)
        )

        self.optimizer = optim.RMSprop(self.eval_net.parameters(), lr=self.lr)
        self.loss_func = nn.MSELoss()

        # Cost history
        self.cost_his = []

    def choose_action(self, observation):
        # 将观察值转换为 PyTorch 张量并添加批次维度
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)

        # 使用评估网络计算动作值
        actions_value = self.eval_net.forward(observation)

        # 将动作值转换为 numpy 数组，便于处理
        actions_value_np = actions_value.detach().numpy()

        # 更新动作值估计
        if not hasattr(self, 'q'):
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q * 0.99 + 0.01 * np.max(actions_value_np)
        self.q.append(self.running_q)

        # Epsilon-贪婪策略
        if np.random.uniform() < self.epsilon:
            # 根据网络输出选择最佳动作
            action = np.argmax(actions_value_np)
        else:
            # 选择随机动作
            action = np.random.randint(0, self.n_actions)

        return action

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # 检查是否需要替换目标网络的参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')

        # 从记忆中随机抽取一批数据
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        # 转换为 PyTorch 张量
        b_s = torch.FloatTensor(batch_memory[:, :self.n_features])
        b_a = torch.LongTensor(batch_memory[:, self.n_features:self.n_features + 1].astype(int))
        b_r = torch.FloatTensor(batch_memory[:, self.n_features + 1:self.n_features + 2])
        b_s_ = torch.FloatTensor(batch_memory[:, -self.n_features:])

        # 获取下一个状态的 Q 值
        q_next = self.target_net(b_s_).detach()  # 从图中分离出来，防止梯度回传
        q_eval = self.eval_net(b_s)

        # 使用双 DQN 选择动作
        if self.double_q:
            q_eval4next = self.eval_net(b_s_).detach()
            max_act4next = q_eval4next.max(1)[1]
            selected_q_next = q_next.gather(1, max_act4next.unsqueeze(1)).squeeze(1)
        else:
            selected_q_next = q_next.max(1)[0]

        q_target = b_r.squeeze(1) + self.gamma * selected_q_next

        # 获取与实际执行动作相对应的 Q 值
        q_eval_wrt_a = q_eval.gather(1, b_a).squeeze(1)

        # 计算损失
        loss = self.loss_func(q_eval_wrt_a, q_target)

        # 优化模型
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 记录成本
        self.cost_his.append(loss.item())

        # 更新 epsilon
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def plot_cost(self, name='RL'):
        plt.figure()
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.title(f'Cost Over Time - {name}')
        plt.savefig(f'./data/{name}_cost.svg', format='svg', dpi=400)
        plt.savefig(f'./data/{name}_cost.png', format='png', dpi=400)
        plt.show()

# Example usage
# dqn = DoubleDQN(n_actions=number_of_actions, n_features=number_of_features)
# Use dqn for training and decision making
