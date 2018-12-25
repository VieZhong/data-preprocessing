# encoding=utf-8
import sys
import os
import json
import shutil
import jieba

jsons_dir = "/data/nssd_data/original_nssd_data"
tokenized_dir = "/data/nssd_data/article_tokenized_json"

def tokenize_stories(file_dir, tokenized_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." % (file_dir, tokenized_dir))
  if not os.path.exists("tmp"): os.makedirs("tmp")

  json_files = os.listdir(file_dir)
  for j_file in json_files:
    lines = read_text_file(os.path.join(file_dir, j_file))
    num_stories = len(lines)
    name = j_file.split('.')[0]

    with open(os.path.join(tokenized_dir, 'kp20k_' + name + '.json'), 'w', encoding='utf-8') as wf:
      for idx, line in enumerate(lines):
        if idx % 1000 == 0:
          print("Writing %s %i of %i; %.2f percent done" % (name, idx, num_stories, float(idx) * 100.0 / float(num_stories)))
        try:
          result = json.loads(line)
        except json.decoder.JSONDecodeError:
          print ("读取错误，直接跳过本行")
          continue
        if "title" not in result or "abstract" not in result or "keyword" not in result:
          continue
        if not len(result['title']) or not len(result['abstract']) or not len(result['keyword']):
          continue
        data = dict()
        data['title'] = ' '.join(jieba.cut(result['title']))
        data['abstract'] = ' '.join(jieba.cut(result['abstract']))
        data['keyword'] = ';'.join([' '.join(jieba.cut(w)) for w in result['keyword'].split(';')])
        wf.write("%s\n" % json.dumps(data))
      wf.close()


def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding='utf-8') as f:
    for line in f:
      lines.append(line.strip())
  return lines


if __name__ == '__main__':

  # Create some new directories
  if not os.path.exists(tokenized_dir): os.makedirs(tokenized_dir)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  tokenize_stories(jsons_dir, tokenized_dir)

