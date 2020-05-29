import os
import sys
import os
import hashlib
import struct
import subprocess
import collections
import tensorflow as tf
from nltk.tokenize import sent_tokenize
from tensorflow.core.example import example_pb2

dm_single_close_quote = u'\u2019' # unicode
dm_double_close_quote = u'\u201d'
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"] # acceptable ways to end a sentence

# We use these to separate the summary sentences in the .bin datafiles
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

#transcript_tokenized_dir = "/datadrive/pmehta/pointer-generator/experiments/transcripts_tokenized"
#finished_files_dir = "/datadrive/pmehta/pointer-generator/experiments/finished_files"
#sentence_per_line_dir = "/datadrive/pmehta/pointer-generator/experiments/sentence_per_line_dir"
#chunks_dir = os.path.join(finished_files_dir, "chunked")

VOCAB_SIZE = 200000
CHUNK_SIZE = 10 # num examples per chunk, for the chunked data

def chunk_file(set_name):
  in_file = '{}/{}.bin'.format(finished_files_dir, set_name)
  reader = open(in_file, "rb")
  chunk = 0
  finished = False
  while not finished:
    chunk_fname = os.path.join(chunks_dir, '%s_%03d.bin' % (set_name, chunk)) # new chunk
    with open(chunk_fname, 'wb') as writer:
      for _ in range(CHUNK_SIZE):
        len_bytes = reader.read(8)
        if not len_bytes:
          finished = True
          break
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
        writer.write(struct.pack('q', str_len))
        writer.write(struct.pack('%ds' % str_len, example_str))
      chunk += 1

def chunk_all():
  # Make a dir to hold the chunks
  if not os.path.isdir(chunks_dir):
    os.mkdir(chunks_dir)
  # Chunk the data
  for set_name in ['test']:
    print("Splitting %s data into chunks..." % set_name)
    chunk_file(set_name)
  print("Saved chunked data in %s" % chunks_dir)

def read_text_file(text_file):
  lines = []
  with open(text_file, "r") as f:
    #fname = '.'.join(text_file.split('.')[0:-1])
    for line in f:
      lines.append(line.strip())
    #lines.append("")
    #lines.append("@highlight")
    #lines.append("")
    #lines.append(fname)
  return lines

def tokenize_stories(stories_dir, tokenized_stories_dir):
  """Maps a whole directory of .story files to a tokenized version using Stanford CoreNLP Tokenizer"""
  print("Preparing to tokenize %s to %s..." % (stories_dir, tokenized_stories_dir))
  stories = os.listdir(stories_dir)
  # make IO list file
  print("Making list of files to tokenize...")
  with open("mapping.txt", "w") as f:
    for s in stories:
      f.write("%s \t %s\n" % (os.path.join(stories_dir, s), os.path.join(tokenized_stories_dir, s)))
  command = ['java', 'edu.stanford.nlp.process.PTBTokenizer', '-ioFileList', '-preserveLines', 'mapping.txt']
  print("Tokenizing %i files in %s and saving in %s..." % (len(stories), stories_dir, tokenized_stories_dir))
  subprocess.call(command)
  print("Stanford CoreNLP Tokenizer has finished.")
  os.remove("mapping.txt")

  # Check that the tokenized stories directory contains the same number of files as the original directory
  num_orig = len(os.listdir(stories_dir))
  num_tokenized = len(os.listdir(tokenized_stories_dir))
  if num_orig != num_tokenized:
    raise Exception("The tokenized stories directory %s contains %i files, but it should contain the same number as %s (which has %i files). Was there an error during tokenization?" % (tokenized_stories_dir, num_tokenized, stories_dir, num_orig))
  print("Successfully finished tokenizing %s to %s.\n" % (stories_dir, tokenized_stories_dir))

def fix_missing_period(line):
  """Adds a period to a line that is missing a period"""
  if "@highlight" in line: return line
  if line=="": return line
  if line[-1] in END_TOKENS: return line
  # print line[-1]
  return line + " ."


def get_art_abs(story_file):
  lines = read_text_file(story_file)

  # Lowercase everything
  lines = [line.lower() for line in lines]

  # Put periods on the ends of lines that are missing them (this is a problem in the dataset because many image captions don't end in periods; consequently they end up in the body of the article as run-on sentences)
  lines = [fix_missing_period(line) for line in lines]

  # Separate out article and abstract sentences
  article_lines = []
  highlights = []
  next_is_highlight = False
  for idx,line in enumerate(lines):
    if line == "":
      continue # empty line
    elif line.startswith("@highlight"):
      next_is_highlight = True
    elif next_is_highlight:
      highlights.append(line)
    else:
      article_lines.append(line)

  # Make article into a single string
  article = ' '.join(article_lines)

  # Make abstract into a signle string, putting <s> and </s> tags around the sentences
  abstract = ' '.join(["%s %s %s" % (SENTENCE_START, sent, SENTENCE_END) for sent in highlights])

  return article, abstract


def write_to_bin(out_file, makevocab=False):
  """Reads the tokenized .story files corresponding to the urls listed in the url_file and writes them to a out_file."""
  print("Making bin file for URLs listed in %s...")
  story_fnames = os.listdir(transcript_tokenized_dir) # TODO: Rewrite
  num_stories = len(story_fnames)

  if makevocab:
    vocab_counter = collections.Counter()

  with open(out_file, 'wb') as writer:
    for idx,s in enumerate(story_fnames):
      if idx % 100 == 0:
        print("Writing story %i of %i; %.2f percent done" % (idx, num_stories, float(idx)*100.0/float(num_stories)))

      # Look in the tokenized story dirs to find the .story file corresponding to this url
      story_file = os.path.join(transcript_tokenized_dir, s)

      # Get the strings to write to .bin file
      article, abstract = get_art_abs(story_file)
      article = article.encode('utf8')
      abstract = abstract.encode('utf8')

      # Write to tf.Example
      tf_example = example_pb2.Example()
      tf_example.features.feature['article'].bytes_list.value.extend([article])
      tf_example.features.feature['abstract'].bytes_list.value.extend([abstract])
      tf_example_str = tf_example.SerializeToString()
      str_len = len(tf_example_str)
      writer.write(struct.pack('q', str_len))
      writer.write(struct.pack('%ds' % str_len, tf_example_str))

      # Write the vocab to file, if applicable
      if makevocab:
        art_tokens = article.split(' ')
        abs_tokens = abstract.split(' ')
        abs_tokens = [t for t in abs_tokens if t not in [SENTENCE_START, SENTENCE_END]] # remove these tags from vocab
        tokens = art_tokens + abs_tokens
        tokens = [t.strip() for t in tokens] # strip
        tokens = [t for t in tokens if t!=""] # remove empty
        vocab_counter.update(tokens)

  print("Finished writing file %s\n" % out_file)

  # write vocab to file
  if makevocab:
    print("Writing vocab file...")
    with open(os.path.join(finished_files_dir, "vocab"), 'w') as writer:
      for word, count in vocab_counter.most_common(VOCAB_SIZE):
        writer.write(word + ' ' + str(count) + '\n')
    print("Finished writing vocab file")

def create_sent_per_line(transcript_text_dir, sentence_per_line_dir):
    for t in os.listdir(transcript_text_dir):
        with open('{}/{}'.format(transcript_text_dir, t)) as f:
            sents = sent_tokenize(f.read())
        text = sents[0]
        for s in sents[1:]:
            text += '\n\n' + s
        fname = '.'.join(t.split('.')[0:-1])
        text += '\n\n' + '@highlight' + '\n\n' + fname
        with open('{}/{}'.format(sentence_per_line_dir,t), 'w') as f:
            f.write(text)

if __name__ == '__main__':
  if len(sys.argv) != 2:
    print("USAGE: python make_datafiles.py <transcript_text_dir>")
    sys.exit()
  transcript_text_dir = sys.argv[1]
  #print("ppp: ", transcript_text_dir)
  transcript_tokenized_dir =  '/'.join(transcript_text_dir.split('/')[0:-1])+'/temp/transcripts_tokenized'
  finished_files_dir = '/'.join(transcript_text_dir.split('/')[0:-1])+"/temp/finished_files"
  sentence_per_line_dir = '/'.join(transcript_text_dir.split('/')[0:-1])+"/temp/sentence_per_line_dir"
  chunks_dir = os.path.join(finished_files_dir, "chunked")

  print(transcript_text_dir, finished_files_dir, sentence_per_line_dir, chunks_dir)

  # Create some new directories
  if not os.path.exists(sentence_per_line_dir): os.makedirs(sentence_per_line_dir)
  if not os.path.exists(transcript_tokenized_dir): os.makedirs(transcript_tokenized_dir)
  if not os.path.exists(finished_files_dir): os.makedirs(finished_files_dir)

  # Create sentence per line file as required input format
  create_sent_per_line(transcript_text_dir, sentence_per_line_dir)

  # Run stanford tokenizer on both stories dirs, outputting to tokenized stories directories
  tokenize_stories(sentence_per_line_dir, transcript_tokenized_dir)

  # Read the tokenized stories, do a little postprocessing then write to bin files
  write_to_bin(os.path.join(finished_files_dir, "test.bin"))

  # Chunk the data. This splits each of train.bin, val.bin and test.bin into smaller chunks, each containing e.g. 1000 examples, and saves them in finished_files/chunks
  chunk_all()
