#! /bin/bash

for i in "control" "drmm" "drmm-only" "knrm" "knrm-only"; do
  echo "building folds for $i"
  python create_training_folds.py --type=$i
done