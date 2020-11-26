#! /bin/bash

RANKER_ARGS=""

for experiment in "control" "drmm" "drmm-only" "knrm" "knrm-only"; do
  for ranker in "LambdaMART" "RandomForest" "AdaRank"; do
    if [ "$ranker" = "LambdaMART" ]; then
      RANKER_ARGS="-ranker 6 -tree 500"
    elif [ "$ranker" = "RandomForest" ]; then
      RANKER_ARGS="-ranker 8 -round 300 -rtype 6"
    elif [ "$ranker" = "AdaRank" ]; then
      RANKER_ARGS="-ranker 3 -round 300"
    else
      exit 1
    fi
    mkdir -p output/${experiment}/${ranker}
    for fold in "f1" "f2" "f3" "f4" "f5"; do
      echo "Beginning to train $ranker on fold $fold"
        java -jar bin/RankLib-2.12.jar ${RANKER_ARGS} -train test-data/${experiment}/${fold}/train.txt -test test-data/${experiment}/${fold}/test.txt -validate test-data/${experiment}/${fold}/valid.txt -metric2t NDCG@10 -metric2T NDCG@10 -gmax 1 -save output/${experiment}/${ranker}/${fold}_model.txt > output/${experiment}/${ranker}/${fold}_meta.txt
    done
  done
done