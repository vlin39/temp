#!/bin/bash

########################################
############# CSCI 2951-O ##############
########################################

# Update this file with instructions on how to compile your code
if [ ! -d "venv" ]; then
    python3 -m venv venv
fi
./venv/bin/pip install -r requirements.txt