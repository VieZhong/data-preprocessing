import sys
import os
import hashlib
import struct
import subprocess
import collections
import json
import tensorflow as tf
import shutil
from tensorflow.core.example import example_pb2


dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

tokenized_dir = "/data/kp20k/validation_json_token/tokenized"
finished_files_dir = "/data/kp20k/validation_json_token/"


def tokenize_stories(file_path, tokenized_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." % (file_path, tokenized_dir))

  if not os.path.exists("tmp"): os.makedirs("tmp")

  lines = read_text_file(file_path)
  names = []
  for line in lines:
    result = json.loads(line)
    file_name = ("%s.txt" % hashhex(result['title']))
    names.append(file_name)
    with open(("tmp/%s" % file_name), "w") as wf:
      # wf.write("%s %s\n" % (result['title'], result['abstract']))
      wf.write("@title\n %s\n" % result['title'])
      wf.write("@abstract\n %s\n" % result['abstract'])
      wf.write("@keyphrases\n %s" % result['keyword'])
      wf.close()

  # make IO list file
  print("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in names:
      f.write("%s \t %s\n" % (os.path.join("tmp", s), os.path.join(tokenized_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-lowerCase', '-ioFileList', '-preserveLines', 'mapping.txt']
  print("Tokenizing %i files in %s and saving in %s..." % (len(names), file_path, tokenized_dir))
  subprocess.call(command)
  print("Stanford CoreNLP Tokenizer has finished.")
  os.remove("mapping.txt")
  
  shutil.rmtree("tmp")

  return names

  # Check that the tokenized stories directory contains the same number of files as the original directory
  # num_orig = len(os.listdir(stories_dir))
  # num_tokenized = len(os.listdir(tokenized_stories_dir))
  # if num_orig != num_tokenized:
  #   raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  # print "Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir)


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
  return line + " ."


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
  num_stories = len(stories)

  with open(out_file, 'w', encoding="utf-8") as writer:
    for s in stories:
      # Look in the tokenized story dirs to find the .story file corresponding to this url
      if os.path.isfile(os.path.join(tokenized_dir, s)):
        story_file = os.path.join(tokenized_dir, s)
      else:
        print("Error: Couldn't find tokenized story file %s in tokenized directories %s. Was there an error during tokenization?" % (s, tokenized_dir))

        raise Exception("Tokenized stories directories %s contain correct number of files but story file %s found in neither." % (tokenized_dir, s))

      # Get the strings to write to .bin file
      title, article, keyword = get_art_abs(story_file)
      # Write
      writer.write("%s\n" % json.dumps({"title": title, "abstract": article, "keyword": keyword}))

  print("Finished writing file %s\n" % out_file)


def check_num_stories(stories_dir, num_expected):
  num_stories = len(os.listdir(stories_dir))
  if num_stories != num_expected:
    raise Exception("stories directory %s contains %i files but should contain %i" % (stories_dir, num_stories, num_expected))


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
