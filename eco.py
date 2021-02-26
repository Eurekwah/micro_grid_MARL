'''
Auther: Eurekwah
Date: 1970-01-01 08:00:00
LastEditors: Eurekwah
LastEditTime: 2021-02-24 13:06:48
FilePath: /code/eco.py
'''
import numpy as np

PEAK  = 0
PLAIN = 1
VALLY = 2

price = [0.6, 0.42, 0.35]
elasticity_coefficient = np.array([[-0.24, 0.12, 0.15], [0.12, -0.18, 0.11], [0.15, 0.11, -0.22]])
print(elasticity_coefficient)

def weight(kind):
    k_temp_1 = 1 + elasticity_coefficient[kind][kind] * (price[kind] - 0.42) / 0.42
