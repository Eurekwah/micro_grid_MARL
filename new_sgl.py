'''
Auther: Eurekwah
Date: 2021-01-29 18:25:09
LastEditors: Eurekwah
LastEditTime: 2021-03-09 03:37:48
FilePath: /code/new_sgl.py
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

    def run(self, action):
        p = (action + 1) / 2 * self.max_output
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
        env_cost = (649 * 0.210 + 0.206 * 14.824 + 9.89 * 62.964) * self.crt_output / 1000
        fuel_cost = 0.206 * self.crt_output
        total_cost = om_cost + env_cost + fuel_cost
        return total_cost

# if __name__ == '__main__':
