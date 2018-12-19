# encoding=utf-8

import sys
import os
import subprocess
import shutil

TAGGER_MODEL_PATH = "/project/stanford-postagger-full-2018-10-16/models/english-caseless-left3words-distsim.tagger"

def tag_article(text):

  with open("text.txt", "w") as f:
    f.write(text)
    f.close()
  command = ['java', 'edu.stanford.nlp.tagger.maxent.MaxentTagger', '-model', TAGGER_MODEL_PATH, '-textFile', 'text.txt', '-tokenize', 'false', '-outputFile', 'tmp.txt']
  subprocess.call(command)
  text_with_tag = read_text_file("tmp.txt")[0]
  os.remove("text.txt")
  os.remove("tmp.txt")
  words_with_tag = text_with_tag.split(" ")
  words = text.split(" ")

  assert len(words_with_tag) == len(words)
  tags = []
  for i, word_with_tag in enumerate(words_with_tag):
    split_index = word_with_tag.rfind("_")
    tags.append(word_with_tag[split_index + 1 :])
  return " ".join(tags)

def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding='utf-8') as f:
    for line in f:
      lines.append(line.strip())
  return lines

print(tag_article("This code produces the non-anonymized version of the CNN / Daily Mail summarization dataset, as used in the ACL 2017 paper Get To The Point: Summarization with Pointer-Generator Networks. It processes the dataset into the binary format expected by the code for the Tensorflow model."))
