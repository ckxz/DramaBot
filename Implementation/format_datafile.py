import re, os
import codecs
import csv

corpus_name = "cornell movie-dialogs corpus"
corpus = os.path.join("/Users/ckxz/Google Drive (ickxzbot@gmail.com)", corpus_name)
formatted_dataset = os.path.join(corpus, "formatted_movie_lines.txt")

# DIALOGUE PREPROCESSING: GET UTTERANCES IN PROPER SHAPE #

# *movie_lines.txt* is structured according to following fields:
MOVIE_LINES_FIELDS = ['lineID', 'characterID', 'movieID', 'character', 'text']

# *movie_conversations.txt* is structured according to following fields:
MOVIE_CONVERSATIONS_FIELDS = ['character1ID', 'character2ID', 'movieID', 'utteranceIDs']


# Returns a dictionary of length 304713 (nº of lines in movie_lines.txt):\
# 	keys are 'lineID's, values are dictionaries of keys MOVIE_LINE_FIELDS with corresponding data values
def loadLines(movie_lines, fields):
	lines = {}
	with open(movie_lines, 'r', encoding='iso-8859-1') as f:
		for line in (f):
			values = line.split(" +++$+++ ")

			# Extract fields
			lineObj = {}
			for i, field in enumerate(fields):
				lineObj[field] = values[i]
			lines[lineObj['lineID']] = lineObj
	return lines


# Matches utterance ids from dialogue sequences from *movie_conversations.txt* (utteranceIDs) with the actual lines (text) from loaded with loadLines().\
# 	Returns list of dictionaries
# e.g.: {'character1ID': 'u0', 'character2ID': 'u2', 'movieID': 'm0', 'utteranceIDs': "['L194', 'L195']\n",\
# 	'lines': [{'lineID': 'L194', 'characterID': 'u0', 'movieID': 'm0', 'character': 'BIANCA', 'text': 'Can we make this quick?  Roxanne Korrine and Andrew Barrett are having an incredibly horrendous public break- up on the quad.  Again.\n'},\
# 		{'lineID': 'L195', 'characterID': 'u2', 'movieID': 'm0', 'character': 'CAMERON', 'text': "Well, I thought we'd start with pronunciation, if that's okay with you.\n"}]}
def loadConversations(fileName, lines, fields):
	conversations = []
	with open(fileName, 'r', encoding='iso-8859-1') as f:
		for line in f:
			values = line.split(" +++$+++ ")
			convObj = {}
			for i, field in enumerate(fields):
				convObj[field] = values[i]
			utterance_id_pattern = re.compile('L[0-9]+')
			# returns list of strings matching the above defined pattern "['L598485', 'L598486', ...]")
			lineIds = utterance_id_pattern.findall(convObj["utteranceIDs"])
			convObj["lines"] = []
			for lineId in lineIds:
				convObj["lines"].append(lines[lineId])
			conversations.append(convObj)
	return conversations


# Pairs up each two sequential utterances from conversation['lines']['text']
def extractSentencePairs(conversations):
	qa_pairs = []
	for conversation in conversations:
		# Iterates over all elements (utterances) in conversation['lines']['text']
		for i in range(len(conversation["lines"]) - 1):
			inputLine = conversation["lines"][i]["text"].strip()
			targetLine = conversation["lines"][i + 1]["text"].strip()
			# Make sure both inputLine and targetLine have values
			if inputLine and targetLine:
				qa_pairs.append([inputLine, targetLine])
	return qa_pairs


# Allows printing n lines from newly formatted file (create_n_datafile())
def printLines(file, n=10):
	with open(file, 'rb') as datafile:
		lines = datafile.readlines()
	for line in lines[:n]:
		print(line)


def create_n_datafile():
	delimiter = '\t'  # tab character
	delimiter = str(codecs.decode(delimiter, 'unicode_escape'))

	print(delimiter)

	lines = {}
	conversations = []
	# print('\nProcessing corpus...')

	# Load dialogue lines with corresponding metadata (MOVIE_LINES_FIELDS)
	lines = loadLines(os.path.join(corpus, 'movie_lines.txt'), MOVIE_LINES_FIELDS)

	# print('\nLoading conversations...')

	conversations = loadConversations(os.path.join(corpus, 'movie_conversations.txt'), lines,
									  MOVIE_CONVERSATIONS_FIELDS)

	# print('\nWriting newly formatted file...')

	# Creates new .csv file with two utterances (question - answer) per row, delimitting utterances with a tab (\t) character
	with open(formatted_dataset, 'w', encoding='utf-8') as outputfile:
		writer = csv.writer(outputfile, delimiter=delimiter, lineterminator='\n')
		for pair in extractSentencePairs(conversations):
			writer.writerow(pair)


# printLines(datafile)
# print('\nDone')

create_n_datafile()