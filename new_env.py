'''
Auther: Eurekwah
Date: 2021-01-28 20:55:42
LastEditors: Eurekwah
LastEditTime: 2021-03-09 03:54:06
FilePath: /code/new_env.py
'''

import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt
import new_load as ld
import new_sl as sl
import new_sgl as sgl
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

    # act结构： 0:t时刻储能装置功率 1：t时刻充电补贴  2:t时刻柴油机功率
    # state结构： 0：t时刻负荷 1：t时刻发电功率 2：储能装置容量  3:t时刻等待队列

    def step(self, time, action):
        # 负荷级 输入经济补贴 输出当前的电动车负荷以及经济收益
        ev_load, ev_profit, crt_price, wait_num = self.load.step(action[1], time)
        if ev_load > self.peak:
            self.peak = ev_load

        # 源荷级 输入储能装置功率 输出源荷级出力，储能装置的状态以及成本
        sl_power, sto_cap, sto_cost = self.source_load.step(time, action[0], ev_load)

        # 源网荷级 输入柴油机功率 返回柴油机真实输出功率以及成本
        self.diesel.run(action[2])
        die_power = self.diesel.crt_output
        die_cost = self.diesel.cost()

        # reward：综合运行成本 主网联络线波动
       
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




