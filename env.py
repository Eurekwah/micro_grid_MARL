'''
Auther: Eurekwah
Date: 2021-01-28 20:55:42
LastEditors: Eurekwah
LastEditTime: 2021-06-10 04:12:34
FilePath: /code/env.py
'''

import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
import load as ld
import sl as sl
import sgl as sgl
import matplotlib.pyplot as plt

# actor_dim = 3
# state_dim = 5

class Env:
    def __init__(self):
        self.load = ld.Load()
        self.source_load = sl.SourceLoad()
        self.diesel = sgl.DieselEngine()
        self.price = 1
        self.peak = 10

    # act structure: 0:t-moment energy storage unit power 1:t-moment charging subsidy 2:t-moment diesel engine power
    # state structure: 0:t-moment load 1:t-moment power generation 2: energy storage capacity 3:t-moment waiting queue

    def step(self, time, action):
        # load level
        ev_load, ev_profit, crt_price, wait_num = self.load.step(action[1], time)
        if ev_load > self.peak:
            self.peak = ev_load

        # source load level
        sl_power, sto_cap, sto_cost = self.source_load.step(time, action[0], ev_load)

        # source grid level
        self.diesel.run(action[2])
        die_power = self.diesel.crt_output
        die_cost = self.diesel.cost()

        # reward
       
        act_power = sl_power + die_power
        total_cost = min(ev_load, act_power) * self.price + ev_profit - sto_cost - die_cost
        if ev_load - act_power > 0:
            punishment = (ev_load - act_power) * 10
        else:
            punishment = (act_power - ev_load) * 0.01
        reward =  total_cost - punishment
        
        if time == 23:
            reward -= self.peak
        
        return np.array([ev_load, act_power, sto_cap, wait_num]), reward, crt_price

    def reset(self):
        self.peak = 10
        self.load.reset()
        self.source_load.reset()
        self.diesel.reset()
        return np.array([0, 0, 0, 0])




