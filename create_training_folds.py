from pathlib import Path
# from <https://github.com/faneshion/DRMM/blob/9d348640ef8a56a8c1f2fa0754fe87d8bb5785bd/NN4IR.cpp>
FOLDS = {
    'f1': {'302', '303', '309', '316', '317', '319', '323', '331', '336', '341', '356', '357', '370', '373', '378', '381', '383', '392', '394', '406', '410', '411', '414', '426', '428', '433', '447', '448', '601', '607', '608', '612', '617', '619', '635', '641', '642', '646', '647', '654', '656', '662', '665', '669', '670', '679', '684', '690', '692', '700'},
    'f2': {'301', '308', '312', '322', '327', '328', '338', '343', '348', '349', '352', '360', '364', '365', '369', '371', '374', '386', '390', '397', '403', '419', '422', '423', '424', '432', '434', '440', '446', '602', '604', '611', '623', '624', '627', '632', '638', '643', '651', '652', '663', '674', '675', '678', '680', '683', '688', '689', '695', '698'},
    'f3': {'306', '307', '313', '321', '324', '326', '334', '347', '351', '354', '358', '361', '362', '363', '376', '380', '382', '396', '404', '413', '415', '417', '427', '436', '437', '439', '444', '445', '449', '450', '603', '605', '606', '614', '620', '622', '626', '628', '631', '637', '644', '648', '661', '664', '666', '671', '677', '685', '687', '693'},
    'f4': {'320', '325', '330', '332', '335', '337', '342', '344', '350', '355', '368', '377', '379', '387', '393', '398', '402', '405', '407', '408', '412', '420', '421', '425', '430', '431', '435', '438', '616', '618', '625', '630', '633', '636', '639', '649', '650', '653', '655', '657', '659', '667', '668', '672', '673', '676', '682', '686', '691', '697'},
    'f5': {'304', '305', '310', '311', '314', '315', '318', '329', '333', '339', '340', '345', '346', '353', '359', '366', '367', '372', '375', '384', '385', '388', '389', '391', '395', '399', '400', '401', '409', '416', '418', '429', '441', '442', '443', '609', '610', '613', '615', '621', '629', '634', '640', '645', '658', '660', '681', '694', '696', '699'}
}
FOLD_GROUPS = {
  'f1': {
    'train': 'f2',
    'test' : 'f1',
    'valid': 'f5'
  },
  'f2': {
    'train': 'f5',
    'test' : 'f2',
    'valid': 'f1'
  },
  'f3': {
    'train': 'f1',
    'test' : 'f3',
    'valid': 'f2'
  },
  'f4': {
    'train': 'f2',
    'test' : 'f4',
    'valid': 'f3'
  },
  'f5': {
    'train': 'f1',
    'test' : 'f5',
    'valid': 'f4'
  }
}

def run():
  label_index = {}
  features_index = {}
  feature_mapping = {}

  print('define feature mapping')
  with open('test-data/robust/example.features.txt') as f:
    line = f.readline()
    _, _, *features = line.split()
    index = 1
    for feature in features:
      id, _ = feature.split(':')
      feature_mapping[id] = index
      index += 1
    print(len(feature_mapping))

  print('loading qrels')
  with open('test-data/robust/qrels') as f:
    for line in f:
      query_id, _, doc_id, label = line.split()
      if query_id not in label_index:
        label_index[query_id] = {}

      label_index[query_id][doc_id] = label

  print('loading features')
  with open('test-data/robust/trec45.features.txt') as f:
    lines = 0
    for line in f:
      lines += 1
      query_id, doc_id, *features = line.split()
      if query_id not in features_index:
        features_index[query_id] = {}

      feature_map = {}
      for feature in features:
        id, weight = feature.split(':')
        feature_map[id] = weight
      complete_features = []
      for cur_feature_id, new_feature_id in sorted(feature_mapping.items()):
        # find feature_id in features (if it exists)
        weight = 0
        if cur_feature_id in feature_map:
          weight = feature_map[cur_feature_id]
        # add feature to complete_features
        complete_features.append(f'{new_feature_id}:{weight}')
      features_index[query_id][doc_id] = complete_features

  def create_feature_file_from_fold(fold, parent_fold, type):
    Path(f'test-data/{parent_fold}').mkdir(parents=True, exist_ok=True)
    with open(f'test-data/{parent_fold}/{type}.txt', 'w') as f:
      for query_id in FOLDS[fold]:
        if query_id in label_index:
          for doc_id in label_index[query_id].keys():
            label = label_index[query_id][doc_id]
            # Not all feature and queries exist in features index
            if query_id in features_index and doc_id in features_index[query_id]:
              features = features_index[query_id][doc_id]
              line = f'{label} qid:{query_id} {" ".join(features)} #docid:{doc_id}\n'
              f.write(line)


  print('create training, testing, and validation files for each fold')
  for fold in FOLD_GROUPS:
    train_group, test_group, valid_group = FOLD_GROUPS[fold].values()
    # Build training file
    create_feature_file_from_fold(train_group, fold, 'train')
    # Build test file
    create_feature_file_from_fold(test_group, fold, 'test')
    # Build validation file
    create_feature_file_from_fold(valid_group, fold, 'valid')


if __name__ == '__main__':
  run()