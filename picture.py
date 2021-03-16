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
tau_stack = []
s_0 = 1
k = 1
for r in np.linspace(0, 1):
    if r < math.exp(-s_0 / k):
        tau = 0
    else:
        tau = (k * math.log(r) + s_0) / s_0
    tau_stack.append(tau)
plt.plot(tau_stack)
plt.show()
