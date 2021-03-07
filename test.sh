#!/bin/bash

nohup python3 exec_test.py > /dev/null 2>&1 &

sleep 2
python3 get_current_state.py 

