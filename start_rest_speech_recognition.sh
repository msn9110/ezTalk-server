#!/bin/bash

nohup python3 spr_service.py >> .temp/rest_log.txt 2>&1 &

./look_rest_log.sh