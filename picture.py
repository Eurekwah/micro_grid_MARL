import matplotlib.pyplot as plt
import numpy as np
import math

# 分时电价曲线
# plt.plot([0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 1.2, 1.2, 1.2, 1.2, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8])
'''
plt.plot([1.2, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.8, 0.8, 1.2, 1.2, 1.2])
plt.show()
'''
# 经济补贴曲线
def test(r):
    s_0 = 1
    k = 1
    length = 2
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
    length = -(tau + (k * math.log(0.75) + s_0) / s_0) * 0.5 + 1
    return length

if __name__ == '__main__':
    stack = []
    for i in np.arange(-0.75, 0.75, 0.05):
        stack.append(test(i))
    plt.plot(np.arange(-0.75, 0.75, 0.05), stack)
    plt.show()
    # print(1-test(-0.7))
'''
a = (np.array([281, 257, 225, 182, 141, 106, 62, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 26, 66, 111, 148, 186, 226, 258]) + np.array([19, 19, 19, 21, 28, 29, 31, 35, 36, 40, 44, 51, 58, 77, 98, 109, 96, 72, 53, 42, 40, 40, 35, 24])) * 5 #+ 500 + 500
plt.plot(a)#
plt.plot([105, 310, 865, 2150, 3580, 4670, 4765, 3780, 2410, 1340, 610, 230, 90, 35, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0])
plt.show()
'''