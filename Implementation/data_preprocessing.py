import re, os
import pickle
import unicodedata

corpus_name = 'Cornell Movie-Dialogs'
# Drive paths:
#datafile = '/content/drive/My Drive/cornell movie-dialogs corpus/formatted_movie_lines.txt'
#save_path = '/content/drive/My Drive/706/data_objects'
# Local paths:
datafile = '/Users/ckxz/Google Drive (ickxzbot@gmail.com)/cornell movie-dialogs corpus/formatted_movie_lines.txt'
save_path = '/Users/ckxz/Google Drive (ickxzbot@gmail.com)/706/data_objects'
# Camber paths:
wd = os.getcwd()
#datafile = os.path.join(wd, 'data_objects/formatted_movie_lines.txt')
#save_path = os.path.join(wd, 'data_objects')


# Special tokens
PAD_token = 0  # Enables padding all utterances to same length
SOS_token = 1  # Start-Of-Sentence token: added at the beginning of each utterance
EOS_token = 2  # End-Of-Sentence token: added at the end of each utterance

# Class object with indexed vocabulary attributes (word2index and index2word to tokenize and untokenize sentences respectively)\
	# word count and total number of different words attributes (word2count and num_words respectively), as well as methods to add words (addSentence and addWord)\
		# and trim irrelevant words (trim)
class Voc:
	def __init__(self, name):
		self.name = name
		self.trimmed = False
		self.word2index = {}
		self.word2count = {}
		self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
		self.num_words = 3  # SOS, EOS, PAD

	def addSentence(self, sentence):
		for word in sentence.split(' '):
			self.addWord(word)

	def addWord(self, word):
		if word not in self.word2index:
			self.word2index[word] = self.num_words
			self.word2count[word] = 1
			self.index2word[self.num_words] = word
			self.num_words += 1
		else:
			self.word2count[word] += 1

	# Remove irrelevant words (appearing less than arbitrary count threshold in data set)
	def trim(self, min_count):
		if self.trimmed:
			return
		self.trimmed = True

		keep_words = []

		for k, v in self.word2count.items():
			if v >= min_count:
				keep_words.append(k)

		print('keep_words {} / {} = {:.4f}'.format(
			len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
		))

		# Reinitialize dictionaries
		self.word2index = {}
		self.word2count = {}
		self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
		self.num_words = 3 # Count default tokens

		for word in keep_words:
			self.addWord(word)

# Max utterance length
MAX_LENGTH = 20


# encoding: from unicode to ASCII
def unicodeToAscii(s):
	return ''.join(
		c for c in unicodedata.normalize('NFD', s)
		if unicodedata.category(c) != 'Mn'
	)

# Convert to ASCII and remove all non-letter chars (except for basic punctuation)
def normalizeString(s):
	s = unicodeToAscii(s.lower().strip())
	s = re.sub(r"([.!?])", r" \1", s)
	s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
	s = re.sub(r"\s+", r" ", s).strip()
	return s

# Returns constructed Voc class and a list of lists, each one containing a pair of utterances (question - answer like)
def readVocs(datafile, corpus_name):
	#print("Reading lines...")
	lines = open(datafile, encoding = 'utf-8').\
		read().strip().split('\n')
	pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
	voc = Voc(corpus_name)
	return voc, pairs

# Returns True if both pair utterances are shorter than MAX_LENGTH
def filterPair(p):
	return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

# Returns pair of utterances if both respect filterPair logic
def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]

# Returns a populated Voc and list of filtered (all utterances shorter than MAX_LENGTH) pairs
def loadPrepareData(corpus_name, datafile):
	#print("Start preparing training data ...")
	voc, pairs = readVocs(datafile, corpus_name)
	#print("Read {!s} sentence pairs".format(len(pairs)))
	pairs = filterPairs(pairs)
	#print("Trimmed to {!s} sentence pairs".format(len(pairs)))
	#print("Counting words...")
	for pair in pairs:
		voc.addSentence(pair[0])
		voc.addSentence(pair[1])
	#print("Counted words:", voc.num_words)
	return voc, pairs


voc, pairs = loadPrepareData(corpus_name, datafile)

with open(save_path + '/pairs.pkl', 'wb') as file:
	pickle.dump(pairs, file)

with open(save_path + '/voc.pkl', 'wb') as file:
	voc.__module__ = 'Voc'
	pickle.dump(voc, file)


