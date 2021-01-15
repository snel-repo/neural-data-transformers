#!/bin/bash

if [[ $# -eq 1 ]]
then
    python -u src/run.py --run-type train --exp-config configs/$1.yaml --clear-only True
else
    echo "Expected args <variant> (ckpt)"
fi