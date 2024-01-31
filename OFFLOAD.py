# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 08:48:54 2019

@author: song

"""

"""
RL offloading training process

When using different DQN algorithms, different packages need to be imported.

基于深度强化学习的任务卸载（offloading）训练过程，主要用于在MEC环境中学习最优的任务卸载策略
"""

import soCoM

# 强化学习算法的导入选项，可以根据需要选择不同的算法
from RL_brainDQN import DeepQNetwork as DQN
#from RL_brainDouble import DoubleDQN as DQN
#from RL_brainDueling import DuelingDQN as DQN
#from RL_brainPrioritizedReplay import DQNPrioritizedReplay as DQN

import simpy
import random


# 设置用户数量为soCoM模块中定义的用户数量
USERS_NUM = soCoM.UN  

# 设置仿真时间、缓冲区大小和每个周期的步数
SIM_TIME = 10000
BUFFER = 500 
LEPI = 500 

# 定义OFFLOADQ类，用于强化学习的任务卸载
class OFFLOADQ(object):
    def __init__(self):
       
        self.name = 'DQN'+'_'+str(soCoM.CD)+'_'+str(USERS_NUM)  # 定义用于记录日志的名称
        self.mec = soCoM.MEC()  # 创建MEC对象
        self.action_space = [str(i) for i in range(USERS_NUM)]  # 定义动作空间
        self.n_actions = 2**USERS_NUM   # 定义动作数量
        self.n_features = 6 # 定义状态特征数量
        self.RL = DQN(self.n_actions, self.n_features,
                      learning_rate=0.01,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=200,
                      memory_size=2000,
                      )
        
        self.done = True    # 初始化完成标志
        self.stepcount = 0  # 初始化步数计数
    
    def reset(self):
        # 重置MEC环境和完成标志
        self.mec.reset()  
        self.done = True
    def printCost(self):
        # 打印学习成本图
        self.RL.plot_cost(self.name)
    def step(self, mec_, observation,env_, WAITING_LEN_):
        # 执行算法步骤
        count = 0
        while True:
            count+=1
            # 如果信道容量不足，暂停一段时间
            if mec_.CHANNEL - mec_.CHANNEL_USED <= 1: 
                mec_.SCORE = -abs(mec_.SCORE)
                yield env_.timeout(mec_.TIMER*mec_.Delta*2)
                continue
            yield env_.timeout(mec_.TIMER*mec_.Delta)

            action = self.RL.choose_action(observation)        # 选择一个行动
            userlist = mec_.randombin(action)   # 根据行动生成用户列表
            channel = mec_.CHANNEL-mec_.CHANNEL_USED    # 计算可用的通道容量

            # 对每个用户执行卸载决策
            for i in range(len(userlist)):
                if userlist[i] == 1:
                    userID = i
                    mec_.offloadOne(env_,userID,sum(userlist),channel)
            
            observation_ = mec_.getstate()  # 获取新的状态
            reward = mec_.SCORE # 获取奖励
            self.RL.store_transition(observation, action, reward, observation_) # 存储这次行动的信息

            # 如果满足条件则执行学习过程
            if (self.stepcount > 40) and (self.stepcount % 4 == 0):
                self.RL.learn()

            # 更新观测值
            observation = observation_
    
    def update(self, RDSEED):
        # 更新和训练模型
        self.reset()
        for episode in range(LEPI):
            self.reset()
            print ("learing episode %d" % (episode))
            random.seed(RDSEED)
            for i in range(USERS_NUM):
                user = soCoM.User(i)
                user.usersetting()
                user.usercreat()
                self.mec.USER_LIST.append(user)
            env_ = simpy.Environment()
            WAITING_LEN_ = simpy.Container(env_, BUFFER, init=len(self.mec.WAITING_LIST))
            
            observation = self.mec.getstate()
            env_.process(self.mec.runremote(env_,WAITING_LEN_))
            env_.process(self.mec.refreshsys(env_,WAITING_LEN_))
            env_.process(self.step(self.mec,observation,env_,WAITING_LEN_))
            env_.run(until=SIM_TIME)
            
            self.stepcount += 1
        self.setpcount = 0
        self.reset()
    
 
    
            
            


