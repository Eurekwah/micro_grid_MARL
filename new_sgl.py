'''
Auther: Eurekwah
Date: 2021-01-29 18:25:09
LastEditors: Eurekwah
LastEditTime: 2021-01-29 18:25:21
FilePath: /ddpg_test/new_sgl.py
'''
import numpy as np
import pandas as pd
import math
from scipy import stats
import matplotlib.pyplot as plt

class DieselEngine:
    def __init__(self):
        self.max_output = 600
        self.min_output = 0
        self.crt_output = 0
        self.max_climb  = 100
        self.min_climb  = -100
        self.k_om       = 0.236 #运维系数

    def run(self, p):
        temp = p
        if p - self.crt_output >= self.max_climb:
            temp = self.crt_output + self.max_climb
        elif self.crt_output - p <= self.min_climb:
            temp = self.crt_output + self.min_climb
        self.crt_output = temp

    def reset(self):
        self.crt_output = 0

    def cost(self):
        om_cost = self.k_om * self.crt_output
        env_cost = (649 * 0.210 + 0.206 * 14.824 + 9.89 * 62.964) * self.crt_output
        fuel_cost = 0.206 * self.crt_output
        total_cost = om_cost + env_cost + fuel_cost
        return total_cost


class ConnectLine:
    def __init__(self):
        self.max_output = 1000
        self.min_output = -600
        self.price      = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8]   # 分时电价
        self.crt_power  = 0           # 正买负卖

    def set_power(self, p):
        self.crt_power = p

    def cost(self, t):
        price_cost = self.crt_power * self.price[t]
        env_cost =  (889 * 0.210 + 1.8 * 14.824 + 1.6 * 62.964) * abs(self.crt_power)
        return price_cost + env_cost

if __name__ == '__main__':
    a = ConnectLine()
    print(a.price)