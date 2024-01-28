import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt


class DeepQNetwork(nn.Module):
    def __init__(self, n_actions, n_features, learning_rate=0.01, reward_decay=0.9,
                 e_greedy=0.9, replace_target_iter=100, memory_size=500,
                 batch_size=32, e_greedy_increment=None):
        super(DeepQNetwork, self).__init__()
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
        # PyTorch模型接受的是torch tensor，因此我们需要先将观测转换成tensor
        observation = torch.unsqueeze(torch.FloatTensor(observation), 0)

        if np.random.uniform() < self.epsilon:
            # 使用模型计算所有动作的Q值
            actions_value = self.eval_net.forward(observation)
            # 选择Q值最大的动作
            action = torch.max(actions_value, 1)[1].data.numpy()[0]  # 或者，action = actions_value.max(1)[1].item() 如果你只需要一个数字
        else:
            # 随机选择一个动作
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
        # Check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
            print('\ntarget_params_replaced\n')

        # Sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        b_s = torch.FloatTensor(batch_memory[:, :self.n_features])
        b_a = torch.LongTensor(batch_memory[:, self.n_features].astype(int))
        b_r = torch.FloatTensor(batch_memory[:, self.n_features + 1])
        b_s_ = torch.FloatTensor(batch_memory[:, -self.n_features:])

        # Q_eval w.r.t the action taken
        q_eval = self.eval_net(b_s).gather(1, b_a.unsqueeze(1)).squeeze(1)
        # Q_next w.r.t the next state
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + self.gamma * q_next.max(1)[0]  # max along the action dimension

        # Reset gradients
        self.optimizer.zero_grad()
        # Calculate loss
        loss = self.loss_func(q_eval, q_target)
        # Backpropagation
        loss.backward()
        # Apply gradients
        self.optimizer.step()

        # Save the cost
        self.cost_his.append(loss.item())

        # Decrease epsilon
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
