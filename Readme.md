<!--
 * @Auther: Eurekwah
 * @Date: 2021-06-09 20:29:00
 * @LastEditors: Eurekwah
 * @LastEditTime: 2021-06-09 20:29:00
 * @FilePath: /code/Readme.md
-->
# Micro Grid - DDPG
## Target 
Use multi agent DDPG for micro grid electric vehicle scheduling.

## Howto
```bash
qsub adf.sh or qub gpu.sh
```
Remember to **change the path and e-mail address** to your own.

## Structure
```
- tf_ddpg.py : the ddpg model
    P.S. there are two kinds of replay buffer(line 203 to 247)
- env.py : the micro grid model
    - load.py : load level(the electirc vehicles)
    - sl.py : source load level(storage unit and new energy)
    - sgl.py : source grid level(diesel engine)
```
## Not completed yet
- Implement the multi-agent model
- Implement different parameters for different vehicles and users
- Implement the random distribution of new energy
- Improve the electrical rationality of the model
- Avoid the model from choosing not to charge the vehicles for high profit
- A more reasonable subsidy strategy
- A more reasonable peak punishment
- Algorithm optimization
- ...



