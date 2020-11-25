

# Train only drmm features
for i in "f1" "f2" "f3" "f4" "f5"; do
  java -jar bin/RankLib-2.12.jar -train test-data/drmm-only/${i}/train.txt -test test-data/drmm-only/${i}/test.txt -validate test-data/drmm-only/${i}/valid.txt -ranker 6 -metric2t NDCG@10 -metric2T NDCG@10 -gmax 1 -save output/drmm-only/${i}_RandomForest_model.txt -round 300 -rtype 6 > output/drmm-only/${i}_RandomForest_meta.txt
  # python2 output/control/${i}_RandomForest_model.txt | dot -Tpng > output/control/${i}_RandomForest_tree.png
done

# Train without drmm features
for i in "f1" "f2" "f3" "f4" "f5"; do
  java -jar bin/RankLib-2.12.jar -train test-data/control/${i}/train.txt -test test-data/control/${i}/test.txt -validate test-data/control/${i}/valid.txt -ranker 6 -metric2t NDCG@10 -metric2T NDCG@10 -gmax 1 -save output/control/${i}_RandomForest_model.txt -round 300 -rtype 6 > output/control/${i}_RandomForest_meta.txt
  # python2 output/control/${i}_RandomForest_model.txt | dot -Tpng > output/control/${i}_RandomForest_tree.png
done

# Train with drmm features
for i in "f1" "f2" "f3" "f4" "f5"; do
  java -jar bin/RankLib-2.12.jar -train test-data/drmm/${i}/train.txt -test test-data/drmm/${i}/test.txt -validate test-data/drmm/${i}/valid.txt -ranker 6 -metric2t NDCG@10 -metric2T NDCG@10 -gmax 1 -save output/drmm/${i}_RandomForest_model.txt -round 300 -rtype 6 > output/drmm/${i}_RandomForest_meta.txt
  # python2 output/drmm/${i}_RandomForest_model.txt | dot -Tpng > output/drmm/${i}_RandomForest_tree.png
done

