#!/bin/bash

# * The evaluation code path has legacy code. Evaluation / analysis is done in analysis scripts.

if [[ $# -eq 2 ]]
then
    python -u src/run.py --run-type eval --exp-config configs/$1.yaml --ckpt-path $1.$2.pth
else
    echo "Expected args <variant> (ckpt)"
fi