### 5.11报告

当前训练结果

<img src="F:\科研\micro_grid\ddpg\code\2.png" alt="2" style="zoom: 50%;" />

每小时最大供电量与无干预负荷对比图

![re1](F:\科研\micro_grid\ddpg\code\re1.png)

新能源供电量与无干预负荷对比图

![re2](F:\科研\micro_grid\ddpg\code\re2.png)

可能存在的问题：新能源供电高峰与用电高峰错位、供电量不足（柴油机发电的延迟性）、参数设置问题导致模型选择不进行任何操作

### DDPG

```python
state dimension = 4
    0：current load at t
    1：current power at t
    2：waiting queue at t
action_dimension = 3
    0: energy storage power at t
    1：charging subsidies at t
    2: diesel engine power at t
action learning rate = 1e-5
critic learning rate = 1e-5
GAMMA = 0.5
REPLACEMENT = [
    dict(name='soft', tau=0.001),
    dict(name='hard', rep_iter_a=600, rep_iter_c=500)
][0]           
MEMORY_CAPACITY = 15000
BATCH_SIZE = 256
variance = 3
variance decline = 0.9999
```

### environment

$$
\begin{array}{l}
punishment^t = \bold{abs}(L^t-P_{sl}^t-P_d^t)\\
reward = E_{load}^t-C_{sl}^t-C_d^t-punishment^t\times 10\\
f=reward
\end{array}
$$

### load level

```python
self.list = [11, 22, 64, 148, 192, 223, 173, 101, 43, 21, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1] #2000 vehicles
self.price = 1
```


$$
\begin{array}{l}
R=action=\frac{R_{p,m}}{R_{0,m}}, action\in (-1,1)	\\
\tau=\left\{
    \begin{array}{ll}
    \frac{k\ln (R)+s_0}{s_0}&R_M<R<1\\
    0&-R_m\leq R\leq R_M\\
    -\frac{k\ln (R)+s_0}{s_0}&-1<R<-R_M
    \end{array}
\right.
\\
length=\left\{
    \begin{array}{ll}
    \bold{int}((1-\tau)\times N_{wait})&R\geq 0\\
    \bold{int}((-\tau)\times N_{wait})&R< 0
    \end{array}
\right.
\\
L^t = N_v^t*15\\
L^{(t+1)\%24} = N_v^t\times 10\\
E_{load}^t=\left\{\begin{array}{ll}L^t\times electricity\_price&t\neq 23\\(L^t-N_c^t\times 50)\times electricity\_price&t=23\end{array}\right.
\end{array}
$$
<img src="F:\科研\micro_grid\ddpg\tau.png" style="zoom:72%;" />

### source load level

```python
self.pv_battery = [281,257,225,182,141,106,62,20,0,0,0,0,0,0,0,0,0,26,66,111,148,186,226,258]     # kW
self.wp_battery = [19,19,19,21,28,29,31,35,36,40,44,51,58,77,98,109,96,72,53,42,40,40,35,24] 
self.energy_sto = StorageUnit(1500)
storage unit:
    self.max_cap    = 1500  # kW
    self.capacity   = capacity
    # >0:discharge;<0:charge
    self.max_output = 500
    self.min_output = -500
    self.s_ev_max   = 0.95
    self.s_ev_min   = 0.25
    self.k_om       = 0.1040 # Operation and maintenance factor(￥/kW)
    self.crt_power  = 0
```

$$
\begin{array}{l}
P^t_s=action^t\times max\_output\ \&\ min\_output<P^t<max\_output , action\in(-1,1)\\
S^t = S^{t-1} - P^t\ \&\ 0<C^t<max\_cap \\
C^t_{sl}=k\times \bold{abs}(C^t)\\
P_{sl}^t=P_{wind}^t+P_{solar}^t+P_s^t

\end{array}
$$

### source grid level

```python
self.max_output = 500
self.min_output = 0
self.crt_output = 0
self.max_climb  = 100
self.min_climb  = -100
self.k_om       = 0.236 #Operation and maintenance factor(￥/kW)
```

$$
\begin{array}{l}
	P_d^t = \left\{
	\begin{array}{ll}
	min\_output&action<0\\
	action\times max\_output&action\geq 0
	\end{array}
	\right.
	action\in(-1,1)
	\\
    and\\ 
    min\_climb\leq P_d^t - P_d^{t-1}\leq max\_climb\\
    C_d^t = C_{om}^t+C_{env}^t+C_{fuel}^t\\
    C_{om}^t=k\_om \times \bold{abs}(P_d^t)\\
    C_{env}^t= \frac{(649 \times 0.210 + 0.206 \times 14.824 + 9.89 \times 62.964) \times P_d^t}{1000}\\
    C_{fuel}^t=0.206\times P_d^t
\end{array}
$$



