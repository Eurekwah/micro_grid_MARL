'''
Auther: Eurekwah
Date: 1970-01-01 08:00:00
LastEditors: Eurekwah
LastEditTime: 2021-02-21 21:18:27
FilePath: /ddpg/new_load.py
'''
import math


class Load:
    def __init__(self):
        self.list = [11, 22, 64, 148, 192, 223, 173, 101, 43, 21, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.wait_num = 0
        self.load = [0 for i in range(24)]
        # self.price = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8]
        self.price = [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2]

    def step(self, action, time):
        r = (action + 1) / 2
        s_0 = 1
        k = 1
        if r < math.exp(-s_0 / k):
            tau = 0
        else:
            tau = (k * math.log(r) + s_0) / s_0
        length = int(tau * self.wait_num + (1 - tau) * self.list[time])
        self.wait_num += self.list[time]
        for i in range(length):
            if time != 23:
                self.load[time] += 5
                self.load[time + 1] += 5
            else:
                self.load[time] += 10
            self.wait_num -= 1
        crt_load = self.load[time]
        self.load[time] = 0
        profit = crt_load * r * self.price[time] 
        if time == 23:
            profit -= 5 * self.wait_num
        return crt_load , profit, (1 - r) * self.price[time]

    def reset(self):
        self.wait_num = 0
        self.load = [0 for i in range(24)]

if __name__ == "__main__":
    a = Load()
    for i in range(24):
        a.step(-1, i)
    print(a.load)