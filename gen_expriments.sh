#!/bin/bash

source "/Users/dodang/workspace/arm-py11-venv/bin/activate"
i=1
for demand_size in {100..200..20};
do
  python gen_traffic.py --topo nsfnet --run 03"$i" --n_demands 5 --demand_size $demand_size \
   --critical_delta_min 0 --critical_delta_max 1 --normal_delta_min 2 --normal_delta_max 3 --critical 0.5
  i=$((i + 1))
done

j=1
for delta in {0..6..1};
do
  python gen_traffic.py --topo nsfnet --run 04"$j" --n_demands 5 --demand_size 150 \
  --critical_delta_min 0 --critical_delta_max 1 --normal_delta_min 2 --normal_delta_max 3 --critical 0.$((2 + delta))
  j=$((j + 1))
done