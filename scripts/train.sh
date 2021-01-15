#!/bin/bash

if [[ $# -eq 1 ]]
then
    python -u src/run.py --run-type train --exp-config configs/$1.yaml
elif [[ $# -eq 2 ]]
then
    python -u src/run.py --run-type train --exp-config configs/$1.yaml --ckpt-path $1.$2.pth
else
    echo "Expected args <variant> (ckpt)"
fi