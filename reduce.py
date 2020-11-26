import csv
import numpy as np
import math
import sys
import os
import time

mus= [-0.9,-0.7,-0.5,-0.3,-0.1,0.1,0.3,0.5,0.7,0.9,1.0]
sigmas= [0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.001]


def calculate_soft_tf(path):
  soft_tf_histogram = [0.] * len(mus)

  with open(path) as f:
    raw_data = list(csv.reader(f, delimiter=' '))
    # trim extra space value from end of row
    data = []
    for row in raw_data:
      if row:
        data.append(row)
    if not data or not data[0]:
      return soft_tf_histogram
    for row in data:
      if row and row[len(row) - 1] == '':
        row.pop(len(row) - 1)

  data = np.array(data, dtype=np.float)

  for _, row in enumerate(data):
    for _, val in enumerate(row):
      for index, (mu, sigma) in enumerate(zip(mus, sigmas)):
        diff = val - mu
        soft_tf_histogram[index] += math.exp(-0.5 * diff * diff / sigma / sigma)

  for i in range(len(soft_tf_histogram)):
    soft_tf_histogram[i] /= np.size(data)

  return soft_tf_histogram

def build_query_index(path):
  index = {}
  with open(path) as f:
    for line in f:
      qid, _, docid, _, _, _ = line.split(' ')
      if qid not in index:
        index[qid] = [-1]
      index[qid].append(docid)
  return index

def feature_string(features):
  result = []
  for index, feature in enumerate(features):
    result.append(f'{index}:{feature}')
  return ' '.join(result)

def build_features(query_path, translation_path, start_qid, end_qid):
  index = build_query_index(query_path)

  for qid in range(start_qid, end_qid):
    num_results = len(index[str(qid)])
    for result_index in range(1, num_results - 1):
      path = os.path.join(translation_path, f'{qid}_{result_index}.')
      title_path = path + 'title'
      body_path = path + 'body'
      title_features = calculate_soft_tf(title_path)
      body_features = calculate_soft_tf(body_path)
      doc_id = index[str(qid)][result_index]
      print(f'{qid} {doc_id} {feature_string(title_features + body_features)}')

if __name__ == '__main__':
  if len(sys.argv) is not 5:
    print('Usage python reduce_translation_mat_to_soft_tf path/to/trec45_indri_retrieval.trec path/to/trec45_trans start_qid end_qid')
    exit(1)

  query_path = sys.argv[1]
  translation_path = sys.argv[2]
  start_qid = int(sys.argv[3])
  end_qid = int(sys.argv[4])
  build_features(query_path, translation_path, start_qid, end_qid)
