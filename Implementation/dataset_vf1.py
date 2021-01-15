import os
import pickle
import torch
import itertools
from torch.utils import data
#import data_preprocessing

# Camber path
wd = os.getcwd()
# Drive path
#data_path ='/content/drive/My Drive/706'
# Local path
#data_path = '/Users/ckxz/Google Drive (ickxzbot@gmail.com)/706'


#voc = pickle.load(open(os.path.join(wd, 'data_objects/voc.pkl'), 'rb'))
#pairs = pickle.load(open(os.path.join(wd, 'data_objects/pairs.pkl'), 'rb'))

PAD_token = 0
SOS_token = 1
EOS_token = 2

class dataset(data.Dataset):

	def __init__(self, voc, pairs):
		self.voc = voc
		self.pairs = pairs

	# Returns a tokenized utterance
	def indexesFromSentence(self, voc, sentence):
		return list([SOS_token] + [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token])

	# Returns a padded utterance (of type list) of length 20
	def padding(self, utterance, fillvalue = PAD_token):
		while len(utterance) < 22:
			utterance.append(fillvalue)
		return utterance

	# Returns mask (of type list) where padded positions are set to False, rest to True
	def binaryMatrix(self, l, value=PAD_token):
		m = []
		for i, token in enumerate(l):
			if token == value:
				m.insert(i, 0)
			else:
				m.insert(i, 1)
		return m

	# Returns tokenized (word2idx) and padded input utterance (of type LongTensor) with its length
	def input_var(self, utterance, voc):
		indexes = self.indexesFromSentence(voc, utterance)
		lengths = torch.tensor([len(indexes) ])
		padded_inputs = self.padding(indexes)
		inputs = torch.LongTensor(padded_inputs)
		return inputs, lengths

	# Returns tokenized (word2idx) and padded target utterance with a mask (pad/no_pad) and utterance length
	def target_var(self, utterance, voc):
		indexes = self.indexesFromSentence(voc, utterance)
		target_len = len(indexes)
		padded_target = self.padding(indexes)
		masks = self.binaryMatrix(padded_target)
		masks = torch.BoolTensor(masks)
		targets = torch.LongTensor(padded_target)
		return targets, masks, target_len

	# Returns padded input and target utterances, target and target len
	def batch2TrainData(self, voc, pairs, idx):
		input_sentence = pairs[idx][0]
		target_sentence = pairs[idx][1]
		inputs, lengths = self.input_var(input_sentence, voc)
		targets, masks, target_len = self.target_var(target_sentence, voc)
		return inputs, lengths, targets, masks, target_len

	def __len__(self):
		return len(list(self.pairs))

	def __getitem__(self, idx):
		inputs, lengths, targets, masks, target_len = self.batch2TrainData(self.voc, self.pairs, idx)
		return idx, inputs, lengths, targets, masks, target_len
