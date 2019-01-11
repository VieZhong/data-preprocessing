import sys
import os
import json
import csv


def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding='utf-8') as f:
    for line in f:
      lines.append(line.strip())
  return lines


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("USAGE: python make_datafiles.py <jsons_dir>")
    sys.exit()
  jsons_dir = sys.argv[1]
  
  lines = read_text_file(jsons_dir)

  train_lines = lines[:20000]
  with open('train.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
      for line in train_lines:
        result = json.loads(line)
        spamwriter.writerow(["chy", result["title"], result["abstract"]])

  test_lines = lines[20000:21200]
  with open('test.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile, delimiter=',')
      for line in train_lines:
        result = json.loads(line)
        spamwriter.writerow(["chy", result["title"], result["abstract"]])