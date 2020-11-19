
for i in "f1" "f2" "f3" "f4" "f5"; do
  java -jar bin/RankLib-2.12.jar -train test-data/${i}/train.txt -test test-data/${i}/test.txt -validate test-data/${i}/valid.txt -ranker 6 -metric2t NDCG@10 -metric2T NDCG@10 -gmax 1 -save output/${i}_LambdaMART_model.txt -tree 500 > output/${i}_LambdaMART_meta.txt
  python2 output/${i}_LambdaMART_model.txt | dot -Tpng > output/${i}_LambdaMART_tree.png
done

