import os
import sys

if len(sys.argv) != 2:
  print("Usage: python3 calculate_weights.py AdaBoost_model.txt")
  exit(1)

with open(sys.argv[1], 'r') as f:
  for i in range(5):
    f.readline()
  file_string = f.readline()
  weight_strings = file_string.split()
  totals = {}
  for string in weight_strings:
    [feature, weight] = string.split(':')
    if totals.get(feature):
      totals[feature] += float(weight)
    else:
      totals[feature] = float(weight)
  sorted = sorted(totals.items(), key=lambda item: item[1])
  sorted.reverse()
  print(sorted[:5])
