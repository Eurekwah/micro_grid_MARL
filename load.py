'''
Auther: Eurekwah
Date: 1970-01-01 08:00:00
LastEditors: Eurekwah
LastEditTime: 2021-02-21 21:18:27
FilePath: /ddpg/new_load.py
'''
import math
import matplotlib.pyplot as plt


class Load:
    def __init__(self):
        self.list = [11, 22, 64, 148, 192, 223, 173, 101, 43, 21, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.wait_num = 0
        self.load = [0 for i in range(24)]
        self.price = 1

    def step(self, action, time):
        r = action * 0.75
        self.wait_num += self.list[time]
        s_0 = 1
        k = 1
        if r >= 0:
            if r < math.exp(-s_0 / k):
                tau = 0
            else:
                tau = (k * math.log(r) + s_0) / s_0
        else:
            if abs(r) < math.exp(-s_0 / k):
                tau = 0
            else:
                tau = -(k * math.log(abs(r)) + s_0) / s_0
        length = int((-(tau + (k * math.log(0.75) + s_0) / s_0) * 0.5 + 1) * self.wait_num)
        for i in range(length):
                self.load[time] += 15
                self.load[min(time + 1, 23)] += 10
                self.wait_num -= 1
        crt_load = self.load[time]
        self.load[time] = 0
        profit = crt_load * self.price
        if time == 23:
            profit -= self.wait_num * 10000
        return crt_load , profit, (1 + r) * self.price, self.wait_num

    def reset(self):
        self.wait_num = 0
        self.load = [0 for i in range(24)]