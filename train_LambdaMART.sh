# Train control
for i in "f1" "f2" "f3" "f4" "f5"; do
  java -jar bin/RankLib-2.12.jar -train test-data/control/${i}/train.txt -test test-data/control/${i}/test.txt -validate test-data/control/${i}/valid.txt -ranker 6 -metric2t NDCG@10 -metric2T NDCG@10 -gmax 1 -save output/control/${i}_LambdaMART_model.txt -tree 500 > output/control/${i}_LambdaMART_meta.txt
  # python2 output/control/${i}_LambdaMART_model.txt | dot -Tpng > output/control/${i}_LambdaMART_tree.png
done

# Train knrm only features
for i in "f1" "f2" "f3" "f4" "f5"; do
  java -jar bin/RankLib-2.12.jar -train test-data/knrm-only/${i}/train.txt -test test-data/knrm-only/${i}/test.txt -validate test-data/knrm-only/${i}/valid.txt -ranker 6 -metric2t NDCG@10 -metric2T NDCG@10 -gmax 1 -save output/knrm-only/${i}_LambdaMART_model.txt -tree 500 > output/knrm-only/${i}_LambdaMART_meta.txt
  # python2 output/control/${i}_LambdaMART_model.txt | dot -Tpng > output/control/${i}_LambdaMART_tree.png
done

# Train knrm features
for i in "f1" "f2" "f3" "f4" "f5"; do
  java -jar bin/RankLib-2.12.jar -train test-data/knrm/${i}/train.txt -test test-data/knrm/${i}/test.txt -validate test-data/knrm/${i}/valid.txt -ranker 6 -metric2t NDCG@10 -metric2T NDCG@10 -gmax 1 -save output/knrm/${i}_LambdaMART_model.txt -tree 500 > output/knrm/${i}_LambdaMART_meta.txt
  # python2 output/control/${i}_LambdaMART_model.txt | dot -Tpng > output/control/${i}_LambdaMART_tree.png
done

# Train drmm only features
for i in "f1" "f2" "f3" "f4" "f5"; do
  java -jar bin/RankLib-2.12.jar -train test-data/drmm-only/${i}/train.txt -test test-data/drmm-only/${i}/test.txt -validate test-data/drmm-only/${i}/valid.txt -ranker 6 -metric2t NDCG@10 -metric2T NDCG@10 -gmax 1 -save output/drmm-only/${i}_LambdaMART_model.txt -tree 500 > output/drmm-only/${i}_LambdaMART_meta.txt
  # python2 output/control/${i}_LambdaMART_model.txt | dot -Tpng > output/control/${i}_LambdaMART_tree.png
done

# Train with drmm features
for i in "f1" "f2" "f3" "f4" "f5"; do
  java -jar bin/RankLib-2.12.jar -train test-data/drmm/${i}/train.txt -test test-data/drmm/${i}/test.txt -validate test-data/drmm/${i}/valid.txt -ranker 6 -metric2t NDCG@10 -metric2T NDCG@10 -gmax 1 -save output/drmm/${i}_LambdaMART_model.txt -tree 500 > output/drmm/${i}_LambdaMART_meta.txt
  # python2 output/drmm/${i}_LambdaMART_model.txt | dot -Tpng > output/drmm/${i}_LambdaMART_tree.png
done

