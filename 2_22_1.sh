#!/bin/bash
#PBS -N micro_grid
#PBS -l nodes=1:ppn=20
#PBS -l walltime=88888:00:00
#PBS -q abn
#PBS -j oe
#PBS -m ae
#PBS -M 741340882@qq.com

source /public/home/zyc20000201/.bashrc

# .bashrc

# Source global definitions
if [ -f /etc/bashrc ]; then
        . /etc/bashrc
fi

# Uncomment the following line if you don't like systemctl's auto-paging feature:
# export SYSTEMD_PAGER=

export PATH=/public/home/keyyd/Python/bin:$PATH
# User specific aliases and functions

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/public/home/zyc20000201/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/public/home/zyc20000201/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/public/home/zyc20000201/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/public/home/zyc20000201/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

conda activate base

source /public/software/profile.d/mpi_mpich-intel-3.2.sh

cd $PBS_O_WORKDIR
NPROCS=`wc -l < $PBS_NODEFILE`

python tf_ddpg.py

