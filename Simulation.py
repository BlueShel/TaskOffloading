# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 15:09:31 2019

@author: song
"""

"""
Simulation experiment

Simulation based on SimPy:  https://simpy.readthedocs.io/en/latest/
SimPy is a process-based discrete-event simulation framework based on standard Python.

用于设置和运行MEC环境下的模拟实验
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import soCoM
from OFFLOAD import OFFLOADQ

import random
import simpy

SIM_TIME = 150000
RANDOM_SEED = 40
RHO = 2
BUFFER = 500


# Simulation of comparative non-RL Experiment
def Simulation(rho, name, function):
    # 设置随机种子以保证实验可重复性
    random.seed(RANDOM_SEED)

    # 创建MEC实例
    mec = soCoM.MEC()
    # 调整任务到达率
    mec.RHO = rho * mec.RHO
    # 为模拟实验命名，通常基于用户数量和其他参数
    name += str(mec.USERS_NUM)

    print("Envronment create!")
    # 创建SimPy环境
    env = simpy.Environment()

    print("User create!")
    # 为每个用户创建实例，并初始化
    for i in range(mec.USERS_NUM):
        user = soCoM.User(i)
        user.usersetting()
        user.usercreat()
        mec.USER_LIST.append(user)

    # 创建一个SimPy容器来模拟等待队列的长度
    WAITING_LEN = simpy.Container(env, BUFFER, init=len(mec.WAITING_LIST))

    # 添加各种过程到SimPy环境中
    # 包括远程运行、系统刷新、日志记录等
    env.process(mec.runremote(env, WAITING_LEN))
    env.process(mec.refreshsys(env, WAITING_LEN, name, 'rho' + str(mec.RHO), 1))

    # 根据函数参数选择不同的卸载策略
    if function == 'offline':
        env.process(mec.offloadOF(env, WAITING_LEN))
    elif function == 'online':
        env.process(mec.offloadOL(env, WAITING_LEN))
    elif function == 'Semi':
        env.process(mec.offloadSe(env, WAITING_LEN))

    # 进程用于记录日志
    env.process(mec.writelog(env, name, 'rho', int(mec.RHO)))

    # 运行模拟直到设定的模拟时间
    env.run(until=SIM_TIME)

    # 写入卸载数据到文件
    mec.writeoffload(name, 'rho', int(mec.RHO))
    # 打印每个用户的信息
    for u in mec.USER_LIST:
        u.userprint()



# Simulation of comparative RL Experiment
def SimulationRL(rho, rl):
    random.seed(RANDOM_SEED)
    mec = soCoM.MEC()
    mec.RHO = rho * mec.RHO

    print("Envronment create!")
    env = simpy.Environment()
    print("User create!")
    for i in range(mec.USERS_NUM):
        user = soCoM.User(i)
        user.usersetting()
        user.usercreat()
        mec.USER_LIST.append(user)

    WAITING_LEN = simpy.Container(env, BUFFER, init=len(mec.WAITING_LIST))
    env.process(mec.runremote(env, WAITING_LEN))
    env.process(mec.refreshsys(env, WAITING_LEN, rl.name, 'rho' + str(mec.RHO), 1))
    env.process(mec.offloadDQ(env, WAITING_LEN, rl))
    env.process(mec.writelog(env, rl.name, 'rho', int(mec.RHO)))
    env.run(until=SIM_TIME)
    mec.writeoffload(rl.name, 'rho', int(mec.RHO))
    for u in mec.USER_LIST:
        u.userprint()


online = 'online' + '_' + str(soCoM.CD) + '_'
Simulation(RHO, online, 'online')

offline = 'offline' + '_' + str(soCoM.CD) + '_'
Simulation(RHO, offline, 'offline')

semi = 'semi' + '_' + str(soCoM.CD) + '_'
Simulation(RHO, semi, 'semi')

##########RL##############
print("BEGIN training!")
rl = OFFLOADQ()
rl.mec.RHO = 4
rl.update(RANDOM_SEED)
rl.printCost()
#####################################
SimulationRL(RHO, rl)
# tf.reset_default_graph()
