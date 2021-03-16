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
# state_dim = 4
GAMMA = 0.4

class Env:
    def __init__(self):
        self.load = ld.Load()
        self.source_load = sl.SourceLoad()
        self.diesel = sgl.DieselEngine()
        #self.price = [1.2, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 1.2, 1.2, 1.2] 
        self.price = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]
        self.last_connect = 0

    # act结构： 0：t时刻充电补贴 1:t时刻储能装置功率 2:t时刻柴油机功率
    # state结构： 0：t时刻负荷 1：t时刻发电功率 2：储能装置容量 3:联络线t时刻功率

    def step(self, time, action):
        # 负荷级 输入经济补贴 输出当前的电动车负荷以及经济收益
        ev_load, ev_profit, crt_price = self.load.step(action[0], time)

        # 源荷级 输入储能装置功率 输出源荷级出力，储能装置的状态以及成本
        sl_power, sto_cap, sto_cost = self.source_load.step(time, action[1], ev_load)

        # 源网荷级 输入柴油机功率 返回柴油机真实输出功率以及成本
        self.diesel.run(action[2])
        die_power = self.diesel.crt_output
        die_cost = self.diesel.cost()

        # 主网联络线：补足电量或者出售多余电量 输出波动以及成本
        connect_power = sl_power + die_power - ev_load
        punishment = 0
        if connect_power > 1000:
            punishment = connect_power - 1000
            connect_power = 1000
        elif connect_power < -600:
            punishment = abs(connect_power) - 600
            connect_power = -600
        if connect_power > 0:
            connect_cost =  (889 * 0.210 + 1.8 * 14.824 + 1.6 * 62.964) * abs(connect_power) / 1000
        else:
            connect_cost = -connect_power * self.price[time] + (889 * 0.210 + 1.8 * 14.824 + 1.6 * 62.964) * abs(connect_power) / 1000
        connect_wave = abs(connect_power - self.last_connect)
        self.last_connect = connect_power

        # reward：综合运行成本 主网联络线波动
        total_cost = ev_profit - sto_cost - die_cost - connect_cost
        reward =  GAMMA * (total_cost) + (1 - GAMMA) * (-connect_wave) - punishment * 10

        return np.array([ev_load, sl_power + die_power, sto_cap, connect_power]), reward

    def reset(self):
        self.load.reset()
        self.source_load.reset()
        self.diesel.reset()
        self.last_connect = 0
        return np.array([0, 0, self.source_load.energy_sto.max_cap, 0])




