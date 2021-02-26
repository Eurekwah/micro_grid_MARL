'''
Auther: Eurekwah
Date: 1970-01-01 08:00:00
LastEditors: Eurekwah
LastEditTime: 2021-02-21 21:18:27
FilePath: /ddpg/new_load.py
'''


class Load:
    def __init__(self):
        self.list = [11, 22, 64, 148, 192, 223, 173, 101, 43, 21, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
        self.wait_num = 0
        self.load = [0 for i in range(24)]
        self.price = [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8]

    def step(self, action, time):
        #print(action)
        decount = action
        self.wait_num += self.list[time]
        if decount < 0:
            length = int(-decount * self.wait_num)
        else:
            length = int(1 - decount * self.wait_num)

        for i in range(length):
            self.load[time] += 5
            self.load[(time + 1) % 24] += 5
            self.wait_num -= 1
        # true_price = decount / 2 + self.price[time]
        crt_load = self.load[time]
        self.load[time] = 0
        profit = crt_load * decount 
        if time == 23:
            profit -= 1 * sum(self.load)
            crt_load += sum(self.load)
        return crt_load , profit

    def reset(self):
        self.wait_num = 0
        self.load = [0 for i in range(24)]

if __name__ == "__main__":
    a = Load()
    for i in range(24):
        a.step(-1, i)
    print(a.load)