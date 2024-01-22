# soCoM

It includes:

- soCoM.py: The system model for soCoM, including definition of the task, user, MEC server, communication model, computing model, and energy consumption model.

- OFFLOAD.py: RL offloading training process.

- RLbrain*.py: RL algorithm of DQN, Dueling DQN, Double DQN, Prioritized replay.

- Simulation.py: run this file for soCoM, creating a simulated environment.

- soCoMM.py, OFFLOADM.py, Simulation-multi.py: Multiple servers senario.

## Required packages
- SimPy:  https://simpy.readthedocs.io/en/latest/
- Pytorch: https://pytorch.org/

## How the code works
- For the soCoM simulation, run the file Simulation.py.

- For changing the numbers of user equipment, change the global variable 'UN' in the file soCoM.py.

- For changing the DQN algorithms, change the import of package in the file OFFLOAD.py.


