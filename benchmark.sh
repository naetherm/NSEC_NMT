#!/bin/bash

python3 ./benchmark.py /local/hdd/exports/data/naetherm/nsec/$1 --cpu --path /local/hdd/exports/pa_output/naetherm/ensec/experiments/$2/checkpoint_best.pt --max-len-b 1024
