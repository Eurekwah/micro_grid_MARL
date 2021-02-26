'''
Auther: Eurekwah
Date: 2021-01-28 20:55:42
LastEditors: Eurekwah
LastEditTime: 2021-02-21 19:51:03
FilePath: /ddpg/new_env.py
'''

import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
import new_load as ld
import new_sl as sl
import matplotlib.pyplot as plt

# actor_dim = 2
# state_dim = 4
GAMMA = 0.5

class Env:
    def __init__(self):
        self.load = ld.Load()
        self.source_load = sl.SourceLoad()
        self.price = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8]
        self.last_connect = 0

    # act结构： 0：t时刻充电补贴 1:t时刻储能装置功率
    # state结构： 0：t时刻负荷 1：t+1时刻电价 2：t时刻发电功率 3：储能装置容量

    def step(self, time, action):
        ev_load, ev_profit = self.load.step(action[0], time)
        sl_power, sto_cap, sto_cost = self.source_load.step(time, action[1], ev_load)
        connect_wave = sl_power - ev_load - self.last_connect
        reward = ev_profit + (sl_power - ev_load) * self.price[time] - sto_cost -  connect_wave
        self.last_connect = connect_wave
        return np.array([ev_load, self.price[(time + 1) % 24], sl_power, sto_cap]), reward

    def reset(self):
        self.load.reset()
        self.source_load.reset()
        return np.array([0, self.price[0], 0, 1000])




