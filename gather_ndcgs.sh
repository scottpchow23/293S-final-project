#! /bin/bash
touch results.txt
for experiment in "control" "drmm" "drmm-only" "knrm" "knrm-only"; do
  echo "$experiment" >> results.txt
  for ranker in "LambdaMART" "RandomForest" "AdaRank"; do
    echo "$ranker" >> results.txt
    for fold in "f1" "f2" "f3" "f4" "f5"; do
      tail -n3 output/${experiment}/${ranker}/${fold}_meta.txt | head -n1 | cut -c23- >> results.txt
    done
  done
done