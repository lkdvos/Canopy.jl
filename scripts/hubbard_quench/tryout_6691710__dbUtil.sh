#!/bin/bash

export DISBATCH_KVSSTCP_HOST=10.250.148.197:45489 PYTHONPATH=/mnt/home/carriero/projects/disBatch/beta/disBatch:${PYTHONPATH}

if [[ $1 == '--mon' ]]
then
    exec /usr/bin/python3 /mnt/home/carriero/projects/disBatch/beta/disBatch/disbatchc/dbMon.py /mnt/home/ldevos/Projects/Canopy.jl/hubbard/scripts/hubbard_quench/tryout_6691710_
elif [[ $1 == '--engine' ]]
then
    exec /usr/bin/python3 /mnt/home/carriero/projects/disBatch/beta/disBatch/disBatch "$@"
else
    exec /usr/bin/python3 /mnt/home/carriero/projects/disBatch/beta/disBatch/disBatch --context /mnt/home/ldevos/Projects/Canopy.jl/hubbard/scripts/hubbard_quench/tryout_6691710__dbUtil.sh "$@" < /dev/null 1> /mnt/home/ldevos/Projects/Canopy.jl/hubbard/scripts/hubbard_quench/tryout_6691710__${BASHPID-$$}_context_launch.log
fi
