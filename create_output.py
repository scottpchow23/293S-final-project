#! /usr/bin/python
import os

configurations = {
  "RandomForest": {
    "ranker": 8,
    "name": "RandomForest"
  },
  "AdaRank": {
    "ranker": 3,
    "name": "AdaRank"
  },
  "LambdaMART": {
    "ranker": 6,
    "name": "LambdaMART"
  },
  "MART": {
    "ranker": 0,
    "name": "MART"
  },
}

for key, config in configurations.items():
  print(config)
  ranker = config["ranker"]
  name = config["name"]
  options = ""
  if name == "RandomForest":
    options = "-round 300 -rtype 6"
  elif name == "AdaRank":
    options = "-round 300"
  elif name == "LambdaMART":
    options = "-tree 500"
  elif name == "MART":
    options = "-tree 500"
  for fold_num in range(1,6):
    print(fold_num)
    command = f"java -jar RankLib-2.12.jar -train MQ2008/Fold{fold_num}/train.txt -test MQ2008/Fold{fold_num}/test.txt -validate MQ2008/Fold{fold_num}/vali.txt -ranker {ranker} -metric2t NDCG@10 -metric2T NDCG@10 -save output/fold{fold_num}_{name}_model.txt {options} > output/fold{fold_num}_{name}_meta.txt"
    print(command)
    os.system(command)
