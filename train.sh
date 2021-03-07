#!/bin/bash

remain="no"
args=""
for arg in $@
do
  if [ $arg == "-r" ]; then
    remain="yes"
  else
    args="$args $arg"
  fi
done

if [ $remain == "no" ]; then
  python3 prepare_dataset.py
fi
echo $args
nohup python3 exec_train.py $args > `pwd`/.temp/logs.txt 2>&1 &
