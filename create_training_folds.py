from pathlib import Path
from enum import Enum
import sys
import time
import click
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
# Only queries 301-450
SMALL_FOLDS = {
    'f1': {'302', '303', '309', '316', '317', '319', '323', '331', '336', '341', '356', '357', '370', '373', '378', '381', '383', '392', '394', '406', '410', '411', '414', '426', '428', '433', '447', '448'},
    'f2': {'301', '308', '312', '322', '327', '328', '338', '343', '348', '349', '352', '360', '364', '365', '369', '371', '374', '386', '390', '397', '403', '419', '422', '423', '424', '432', '434', '440', '446'},
    'f3': {'306', '307', '313', '321', '324', '326', '334', '347', '351', '354', '358', '361', '362', '363', '376', '380', '382', '396', '404', '413', '415', '417', '427', '436', '437', '439', '444', '445', '449', '450'},
    'f4': {'320', '325', '330', '332', '335', '337', '342', '344', '350', '355', '368', '377', '379', '387', '393', '398', '402', '405', '407', '408', '412', '420', '421', '425', '430', '431', '435', '438'},
    'f5': {'304', '305', '310', '311', '314', '315', '318', '329', '333', '339', '340', '345', '346', '353', '359', '366', '367', '372', '375', '384', '385', '388', '389', '391', '395', '399', '400', '401', '409', '416', '418', '429', '441', '442', '443'}
}

# 1-24 are typical tfidf features
# 25-213 are drmm features
# 214-241 are aggregate tfidf features
# 242-263 are knrm features
TFIDF_FEATURES = range(25)
DRMM_FEATURES = range(25, 214)
AGGREGATE_FEATURES = range(214, 242)

# Block lists for different experiments
CONTROL_BLOCK_LIST = DRMM_FEATURES
DRMM_ONLY_BLOCK_LIST = [*TFIDF_FEATURES, *AGGREGATE_FEATURES]
DRMM_BLOCK_LIST = {}

experiments = {
  'drmm': {
    'use_drmm_control': True,
    'block_list': DRMM_BLOCK_LIST,
    'use_knrm': False,
  },
  'control': {
    'use_drmm_control': True,
    'block_list': CONTROL_BLOCK_LIST,
    'use_knrm': False,
  },
  'drmm-only': {
    'use_drmm_control': True,
    'block_list': DRMM_ONLY_BLOCK_LIST,
    'use_knrm': False,
  },
  'knrm': {
    'use_drmm_control': True,
    'block_list': CONTROL_BLOCK_LIST,
    'use_knrm': True,
  },
  'knrm-only': {
    'use_drmm_control': False,
    'block_list': [*CONTROL_BLOCK_LIST] + DRMM_ONLY_BLOCK_LIST,
    'use_knrm': True,
  }
}

@click.command()
@click.option('--type', help='Choose an experiment type from: control, drmm, drmm-only, knrm, or knrm-only.', required=True)
def run(type):
  experiment = experiments[type]
  label_index = {}
  drmm_feature_mapping = {}
  drmm_features_index = {}
  knrm_feature_mapping = {}
  knrm_features_index = {}

  feature_block_list = experiment['block_list']

  # The index is shared across features to prevent numbering conflicts between features
  index = 1
  click.echo('defining control + drmm feature mapping')
  with open('test-data/robust/example.features.txt') as f:
    line = f.readline()
    _, _, *features = line.split()
    for feature in features:
      id, _ = feature.split(':')
      drmm_feature_mapping[id] = index
      index += 1

  click.echo('defining knrm feature mapping')
  with open('test-data/robust/knrm_features.csv') as f:
    line = f.readline()
    _, _, *features = line.split()
    for feature in features:
      id, _ = feature.split(':')
      knrm_feature_mapping[id] = index
      index += 1

  click.echo('loading qrels')
  with open('test-data/robust/qrels') as f:
    for line in f:
      query_id, _, doc_id, label = line.split()
      if query_id not in label_index:
        label_index[query_id] = {}

      label_index[query_id][doc_id] = label

  if experiment['use_drmm_control']:
    start = time.time()
    with open('test-data/robust/trec45.features.txt') as f:
      with click.progressbar(f, label='loading control + drmm features') as bar:
        for line in bar:
          query_id, doc_id, *features = line.split()
          if query_id not in drmm_features_index:
            drmm_features_index[query_id] = {}

          feature_weight_map = {}
          for feature in features:
            id, weight = feature.split(':')
            feature_weight_map[id] = weight
          complete_features = []
          for cur_feature_id, new_feature_id in sorted(drmm_feature_mapping.items()):
            # find feature_id in features (if it exists)
            weight = 0
            if cur_feature_id in feature_weight_map:
              weight = feature_weight_map[cur_feature_id] if new_feature_id not in feature_block_list else 0
            # add feature to complete_features
            complete_features.append(f'{new_feature_id}:{weight}')
          drmm_features_index[query_id][doc_id] = complete_features
    end = time.time()
    click.echo(f'Done in {end - start} seconds.')

  if experiment['use_knrm']:
    start = time.time()
    with open('test-data/robust/knrm.features') as f:
      with click.progressbar(f, label='loading knrm features') as bar:
        for line in bar:
          query_id, doc_id, *features = line.split()
          if query_id not in knrm_features_index:
            knrm_features_index[query_id] = {}
          complete_features = []
          for feature in features:
            feature_id, feature_weight = feature.split(':')
            new_feature_id = knrm_feature_mapping[feature_id]
            complete_features.append(f'{new_feature_id}:{feature_weight}')
          knrm_features_index[query_id][doc_id] = complete_features
    end = time.time()
    click.echo(f'Done in {end - start} seconds.')

  def create_feature_file_from_fold(fold, parent_fold, type):
    path_to_folds = f'test-data/{type}'
    Path(f'{path_to_folds}/{parent_fold}').mkdir(parents=True, exist_ok=True)
    with open(f'{path_to_folds}/{parent_fold}/{type}.txt', 'w') as f:
      for query_id in SMALL_FOLDS[fold]:
        if query_id in label_index:
          for doc_id in label_index[query_id].keys():
            label = label_index[query_id][doc_id]
            # Not all feature and queries exist in features index
            drmm_features_exist = query_id in drmm_features_index and doc_id in drmm_features_index[query_id] or not experiment['use_drmm_control']
            knrm_features_exist = query_id in knrm_features_index and doc_id in knrm_features_index[query_id] or not experiment['use_knrm']

            if drmm_features_exist and knrm_features_exist:
              features = []
              if experiment['use_drmm_control']:
                features.extend(drmm_features_index[query_id][doc_id])
              if experiment['use_knrm']:
                features.extend(knrm_features_index[query_id][doc_id])
              line = f'{label} qid:{query_id} {" ".join(features)} #docid:{doc_id}\n'
              f.write(line)

  start = time.time()
  with click.progressbar(FOLD_GROUPS, label='create training, testing, and validation files for each fold') as bar:
    for fold in bar:
      train_group, test_group, valid_group = FOLD_GROUPS[fold].values()
      # Build training file
      create_feature_file_from_fold(train_group, fold, 'train')
      # Build test file
      create_feature_file_from_fold(test_group, fold, 'test')
      # Build validation file
      create_feature_file_from_fold(valid_group, fold, 'valid')
  end = time.time()
  click.echo(f'Done in {end - start} seconds.')


if __name__ == '__main__':
  run()