'''
Auther: Eurekwah
Date: 2021-01-29 17:20:34
LastEditors: Eurekwah
LastEditTime: 2021-02-21 01:20:09
FilePath: /ddpg/new_sl.py
'''
# 储能装置 真实输出功率 剩余电量 为电网净负荷

class StorageUnit:
    def __init__(self, capacity):
        self.max_cap    = 1000  # kW
        self.capacity   = capacity
        # 正放电负充电
        self.max_output = 250
        self.min_output = -250
        self.s_ev_max   = 0.95
        self.s_ev_min   = 0.25
        self.k_om       = 0.1040 # Operation and maintenance factor(￥/kW)
        self.crt_power  = 0

    def step(self, action):
        self.crt_power = action * self.max_output
        self.capacity -= self.crt_power
        if self.capacity >= self.max_cap * self.s_ev_max:
            self.capacity = self.max_cap * self.s_ev_max
        elif self.capacity <= self.max_cap * self.s_ev_min:
            self.capacity = self.max_cap * self.s_ev_min
        temp = self.cost()
        return self.crt_power, self.capacity, temp


    def cost(self):
        # 成本：系数✖功率>0
        return abs(self.crt_power) * self.k_om 

    def reset(self):
        self.capacity  = self.max_cap
        self.crt_power = 0



class SourceLoad:
    def __init__(self):
        self.pv_battery    = [281, 257, 225, 182, 141, 106, 62, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 66, 111, 148, 186, 226, 258] * 5    # kW
        self.wp_battery    = [19, 19, 19, 21, 28, 29, 31, 35, 36, 40, 44, 51, 58, 77, 98, 109, 96, 72, 53, 42, 40, 40, 35, 24] * 5
        self.energy_sto    = StorageUnit(500)

    def step(self, time, action, other_load):
        sto_power, sto_cap, sto_cost = self.energy_sto.step(action)
        sl_power = self.pv_battery[time] + self.wp_battery[time] + sto_power
        return sl_power, sto_cap, sto_cost


    def net_load(self, other_load):
        return self.energy_sto.crt_power + other_load - self.pv_battery - self.wp_battery 
        # -光伏-风力+储能+电动车+其他用电
        # 使传统能源-新能源功率最小

    def reset(self):
        self.energy_sto.reset()
        


if __name__ == '__main__':
    instance = SourceLoad()