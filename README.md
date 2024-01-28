# soCoM

It includes:

- soCoM.py: The system model for soCoM, including definition of the task, user, MEC server, communication model, computing model, and energy consumption model.

- OFFLOAD.py: RL offloading training process.

- RLbrain*.py: RL algorithm of DQN, Dueling DQN, Double DQN, Prioritized replay.

- Simulation.py: run this file for soCoM, creating a simulated environment.

- soCoMM.py:里面三个类（MEC、Job、User）的作用如下：  
Job 类：这个类代表一个工作任务。它包含了任务的基本信息，如任务ID、用户ID、任务类型、任务状态等。此外，它还包含了任务的运行时间、传输时间、计算时间、能耗等信息。这个类的主要作用是模拟和管理单个工作任务的生命周期。  
User 类：这个类代表一个用户。它包含了用户的基本信息，如用户ID、任务列表等。此外，它还包含了用户的任务参数，如任务数量、数据量、计算量、传输时间、计算能耗等。这个类的主要作用是模拟和管理用户的行为，包括创建任务、发送任务、本地运行任务等。  
MEC 类：这个类代表一个移动边缘计算（Mobile Edge Computing，MEC）服务器。它包含了服务器的基本信息，如用户数量、总带宽、CPU使用率等。此外，它还包含了服务器的任务池、传输池、等待列表等。这个类的主要作用是模拟和管理MEC服务器的行为，包括任务卸载、远程运行任务、刷新系统状态等
- OFFLOADM.py, Simulation-multi.py: Multiple servers senario.

## Required packages
- SimPy:  https://simpy.readthedocs.io/en/latest/
- Pytorch: https://pytorch.org/

## How the code works
- For the soCoM simulation, run the file Simulation.py.

- For changing the numbers of user equipment, change the global variable 'UN' in the file soCoM.py.

- For changing the DQN algorithms, change the import of package in the file OFFLOAD.py.


