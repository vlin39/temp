#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################
E_BADARGS=65
if [ $# -lt 1 ] || [ $# -gt 2 ]
then
	echo "Usage: `basename $0` <input> [timeLimit]"
	exit $E_BADARGS
fi
	
input=$1
timeFlag=""
if [ $# -eq 2 ]; then
    timeFlag="--time-limit $2"
fi

if [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
else
    PYTHON="python3"
fi

# run the solver
$PYTHON src/main.py $input $timeFlag