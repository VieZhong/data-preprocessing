# encoding=utf-8
import sys
import os
import hashlib
import struct
import subprocess
import collections
import json
import tensorflow as tf
import shutil
import jieba
from tensorflow.core.example import example_pb2

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

tokenized_dir = "/data/nssd_data/validation_json_token/tokenized"
finished_files_dir = "/data/nssd_data/validation_json_token/"

def tokenize_stories(file_path, tokenized_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." % (file_path, tokenized_dir))

  if not os.path.exists("tmp"): os.makedirs("tmp")

  lines = read_text_file(file_path)
  names = []
  for line in lines:
    try:
      result = json.loads(line)
    except json.decoder.JSONDecodeError:
      print ("读取错误，直接跳过本行")
      continue
    if "title" not in result or "abstract" not in result or "keyword" not in result:
      continue
    if not len(result['title']) or not len(result['abstract']) or not len(result['keyword']):
      continue
    file_name = ("%s.txt" % hashhex(result['title']))
    split_word_title = ' '.join(jieba.cut(result['title']))
    split_word_abstract = ' '.join(jieba.cut(result['abstract']))
    split_word_keyword = ';'.join([' '.join(jieba.cut(w)) for w in result['keyword'].split(';')])
    names.append(file_name)
    with open(("tmp/%s" % file_name), "w") as wf:
      wf.write("@title\n %s\n" % split_word_title)
      wf.write("@abstract\n %s\n" % split_word_abstract)
      wf.write("@keyphrases\n %s" % split_word_keyword)
      wf.close()

    # make IO list file
    print("Making list of files to tokenize...")
    with open("mapping.txt", "w") as f:
      for s in names:
        f.write("%s \t %s\n" % (os.path.join("tmp", s), os.path.join(tokenized_dir, s)))
    command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-lowerCase', '-ioFileList', '-preserveLines', 'mapping.txt']
    print("Tokenizing %i files in %s/%s and saving in %s..." % (len(names), file_dir, j_file, tokenized_dir))
    subprocess.call(command)
    print("Stanford CoreNLP Tokenizer has finished.")
    os.remove("mapping.txt")
  
  shutil.rmtree("tmp")

  return names


def read_text_file(text_file):
  lines = []
  with open(text_file, "r", encoding='utf-8') as f:
    for line in f:
      lines.append(line.strip())
  return lines


def hashhex(s):
  """Returns a heximal formated SHA1 hash of the input string."""
  h = hashlib.sha1()
  h.update(s.encode('utf-8'))
  return h.hexdigest()


def get_url_hashes(url_list):
  return [hashhex(url) for url in url_list]


def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@title" in line: return line
  if "@abstract" in line: return line
  if "@keyphrases" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " 。"


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  # lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  keyphrases = None
  next_is = {
    "title": False,
    "abstract": False,
    "keyphrase": False
  }
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@keyphrases"):
      next_is["keyphrase"] = True
    elif next_is["keyphrase"]:
      keyphrases = line.split(';')
      next_is["keyphrase"] = False
    elif line.startswith("@title"):
      next_is["title"] = True
    elif next_is["title"]:
      title = line
      next_is["title"] = False
    elif line.startswith("@abstract"):
      next_is["abstract"] = True
    elif next_is["abstract"]:
      next_is["abstract"] = False
      article_lines.append(fix_missing_period(line))

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  keyword = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in keyphrases])

  return title, article, keyword


def write_to_json(stories, out_file):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  # print "Making bin file for URLs listed in %s..." % url_file
  # url_list = read_text_file(url_file)
  # url_hashes = get_url_hashes(url_list)
  # story_fnames = [s+".txt" for s in stories]
  num_stories = len(stories)

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'w', encoding="utf-8") as writer:
    for idx,s in enumerate(stories):
      if idx % 1000 == 0:
        print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(tokenized_dir, s)):
        story_file = os.path.join(tokenized_dir, s)
      else:
        print("Error: Couldn't find tokenized story file %s in tokenized directories %s. Was there an error during tokenization?" % (s, tokenized_dir))
        # Check again if tokenized stories directories contain correct number of files
        # print "Checking that the tokenized stories directories %s and %s contain correct number of files..." % (cnn_tokenized_stories_dir, dm_tokenized_stories_dir)
        # check_num_stories(cnn_tokenized_stories_dir, num_expected_cnn_stories)
        # check_num_stories(dm_tokenized_stories_dir, num_expected_dm_stories)
        raise Exception("Tokenized stories directories %s contain correct number of files but story file %s found in neither." % (tokenized_dir, s))

      # Get the strings to write to .bin file
      title, article, keyword = get_art_abs(story_file)
      # Write
      writer.write("%s\n" % json.dumps({"title": title, "abstract": article, "keyword": keyword}))


  print("Finished writing file %s\n" % out_file)


if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("USAGE: python make_datafiles.py <json_path>")
    sys.exit()
  json_path = sys.argv[1]

  # Create some new directories
  if not os.path.exists(tokenized_dir): os.makedirs(tokenized_dir)
  # if not os.path.exists(dm_tokenized_stories_dir): os.makedirs(dm_tokenized_stories_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  stories = tokenize_stories(json_path, tokenized_dir)
  # tokenize_stories(dm_stories_dir, dm_tokenized_stories_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_json(stories, os.path.join(finished_files_dir, "validation.json"))
