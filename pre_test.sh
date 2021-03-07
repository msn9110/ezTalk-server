#!/bin/bash
export testing_set=/home/dmcl/dataset/withTone/_no_tone/_clip_0.8
python3 waveTools.py -ig -p /home/dmcl/dataset/withTone -t 0.8
python3 label_wav_syllable_test.py -s pre_test  -ts ${testing_set} --step

