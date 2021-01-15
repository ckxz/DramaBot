import re, os
import pickle
import itertools
import unicodedata
import codecs
import csv
import random
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
from tensorboardX import SummaryWriter
import dataset_vf1


# Working Directory, Experiment Name
wd = os.getcwd()
xp_name = 'RNN_1'

# ============================== Get device right ============================== #
device = torch.device('cpu')
if torch.cuda.is_available():
	device = torch.device('cuda')

# =============================== Data and paths =============================== #

#voc = pickle.load(open(os.path.join(wd, 'data_objects/voc.pkl'), 'rb'))
#pairs = pickle.load(open(os.path.join(wd, 'data_objects/voc.pkl'), 'rb'))


# =========================== Create dataset and voc =========================== #
corpus_name = 'Cornell Movie-Dialogs'
datafile = os.path.join(wd, 'data_objects/formatted_movie_lines.txt')

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
		self.num_words = 3  # Count default tokens

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
	# print("Reading lines...")
	lines = open(datafile, encoding='utf-8'). \
		read().strip().split('\n')
	pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
	voc = Voc(corpus_name)
	return voc, pairs

# Returns True if both pair utterances are shorter than MAX_LENGTH
def filterPair(p):
	return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH

def filterPairs(pairs):
	return [pair for pair in pairs if filterPair(pair)]

# Returns a populated Voc and list of filtered (all utterances shorter than MAX_LENGTH) pairs
def loadPrepareData(corpus_name, datafile):
	# print("Start preparing training data ...")
	voc, pairs = readVocs(datafile, corpus_name)
	# print("Read {!s} sentence pairs".format(len(pairs)))
	pairs = filterPairs(pairs)
	# print("Trimmed to {!s} sentence pairs".format(len(pairs)))
	# print("Counting words...")
	for pair in pairs:
		voc.addSentence(pair[0])
		voc.addSentence(pair[1])
	# print("Counted words:", voc.num_words)
	return voc, pairs


voc, pairs = loadPrepareData(corpus_name, datafile)

# =================================  Modules =================================== #

# Encoder based on a Gated Recurrent Unit
class Encoder(nn.Module):
	def __init__(self, emb_dim, embedding, hidden_size, n_layers = 1, dropout = 0):
		super(Encoder, self).__init__()
		#self.emb_dim = emb_dim
		self.hidden_size = hidden_size
		self.n_layers = n_layers
		self.wte = embedding
		self.gru = nn.GRU(emb_dim, hidden_size, n_layers, batch_first = False,
						  dropout=(0 if n_layers == 1 else dropout), bidirectional=True)
		self.init_params()

	def init_params(self):
		for x in self.named_parameters():
			x[1].requires_grad = True
			if 'weight' in x[0]:
				torch.nn.init.xavier_uniform_(x[1])
			elif 'bias' in x[0]:
				x[1].data.fill_(0.01)

	def forward(self, input_seq, hidden = None):
		# embed input sequence
		embedded = self.wte(input_seq)
		# Forward-pass
		outputs, hidden = self.gru(embedded, hidden)
		# Sums bidirectional outputs from GRU
		outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
		return outputs, hidden

# Luong's Attention
class LuongAttention(nn.Module):
	def __init__(self, score_method, hidden_size, device = 'cpu'):
		super(LuongAttention, self).__init__()
		self.score_method = score_method
		if self.score_method not in ['dot', 'general', 'concat']:
			raise ValueError(self.score_method, 'is not an appropriate attention method')
		if self.score_method == 'general':
			self.attn = nn.Linear(hidden_size, hidden_size)
		elif self.score_method == 'concat':
			self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
			self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))
		if torch.cuda.is_available():
			self.device = torch.device('cuda')

	def dot_score(self, hidden, encoder_output):
		return torch.sum(hidden * encoder_output, dim = 2)

	def general_score(self, hidden, encoder_output):
		energy = self.attn(encoder_output)
		return torch.sum(hidden * energy, dim = 2)

	def concat_score(self, hidden, encoder_output):
		energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), 1, -1), encoder_output), 2)).tanh()
		return torch.sum(self.other * energy, dim = 2)

	def forward(self, hidden, encoder_output):
		# Compute attention "energies" according to chosen mathematical model
		if self.score_method == 'dot':
			energies = self.dot_score(hidden, encoder_output)
		elif self.score_method == 'general':
			energies = self.general_score(hidden, encoder_output)
		elif self.score_method == 'concat':
			energies = self.concat_score(hidden, encoder_output)
		energies = energies.t()
		return F.softmax(energies, dim = 1).unsqueeze(1) # [batch_size, 1, max_length]


# "Attentive" GRU Decoder
class Decoder(nn.Module):
	def __init__(self, vocab_size, emb_dim, embedding, hidden_size, attn_model, n_layers=1, dropout=0.1):
		super(Decoder, self).__init__()

		# self.emb_dim = emb_dim
		# self.vocab_size = vocab_size
		self.n_layers = n_layers
		# self.dropout = dropout
		self.wte = embedding
		self.embedding_dropout = nn.Dropout(dropout)
		self.gru = nn.GRU(emb_dim, hidden_size, n_layers, batch_first=False,
						  dropout=(0 if n_layers == 1 else dropout))
		self.concat = nn.Linear(hidden_size * 2, hidden_size)
		self.out = nn.Linear(hidden_size, vocab_size)

		self.attn = LuongAttention(attn_model, emb_dim)

		self.init_params()

	def init_params(self):
		for x in self.named_parameters():
			x[1].requires_grad = True
			if 'weight' in x[0]:
				torch.nn.init.xavier_uniform_(x[1])
			elif 'bias' in x[0]:
				x[1].data.fill_(0.01)

	def forward(self, input_step, encoder_outputs, encoder_hidden):
		embedded = self.wte(input_step)
		embedded = self.embedding_dropout(embedded)
		# Forward-pas through GRU
		rnn_output, hidden = self.gru(embedded, encoder_hidden)
		# Compute attention's weights given encoder's and decoder's GRU outputs
		attn_weights = self.attn(rnn_output, encoder_outputs)
		# Build context (what parts from encoder's output to focus on)
		context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
		# Add computed context to decoder's GRU output
		rnn_output = rnn_output.squeeze(0)
		context = context.squeeze(1)
		concat_input = torch.cat((rnn_output, context), 1)
		concat_output = torch.tanh(self.concat(concat_input))
		# Assign probabilities as next token to each of vocab's words
		output = self.out(concat_output)
		output = F.softmax(output, dim=1)

		return output, hidden


# ======================== Define and instantiate model ======================== #
epoch = 0

# Get device right
device = 'cpu'
if (torch.cuda.is_available()):
	device = torch.device('cuda')

epochs = 100
batch_size = 64

# DataLoader
ds = dataset_vf1.dataset(voc, pairs)
dataloader = data.DataLoader(ds, batch_size = batch_size, shuffle=False)

# Encoder & Decoder params:
hidden_size = 256
emb_dim = 30
embedding = nn.Embedding(voc.num_words, emb_dim)
encoder_n_layers = 2
decoder_n_layers = 2
dropout = 0.1
lr = 0.0001
lr_str = '0001'
decoder_learning_ratio = 5.0
clip = 50

# Teacher forcing?
teacher_forcing_ratio = 1.0

# Attention method:
score_method = 'dot'
#score_method = 'general'
#score_method = 'concat'

# Encoder:
encoder = Encoder(emb_dim = emb_dim, embedding = embedding, hidden_size = hidden_size, n_layers = encoder_n_layers, dropout = dropout)
encoder_optimizer = optim.Adam(encoder.parameters(), lr = lr)
encoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(encoder_optimizer, 'min', factor=0.5, patience=5)


#Decoder:
decoder = Decoder(vocab_size = voc.num_words, emb_dim = emb_dim, embedding = embedding, hidden_size = hidden_size, attn_model = score_method, n_layers = decoder_n_layers, dropout = dropout)
decoder_optimizer = optim.Adam(decoder.parameters(), lr = lr * decoder_learning_ratio)
decoder_scheduler = optim.lr_scheduler.ReduceLROnPlateau(decoder_optimizer, 'min', factor=0.5, patience=5)


# Continue experiment? Load model.state_dict() and optimizer.state_dict()
folder_name = os.path.join(wd, xp_name)


if os.path.exists(folder_name):
	try:
		checkpoint = torch.load(os.path.join(folder_name, 'checkpoint.pth'))
	except:
		raise FileNotFoundError("Unable to load checkpoint")
	epoch = checkpoint['epoch'] + 1
	encoder.load_state_dict(checkpoint['encoder'])
	encoder_optimizer.load_state_dict(checkpoint['encoder_optim'])
	decoder.load_state_dict(checkpoint['decoder'])
	decoder_optimizer.load_state_dict(checkpoint['decoder_optim'])
	embedding.load_state_dict(checkpoint['embedding'])
else:
	os.mkdir(folder_name)

encoder = encoder.to(device)
encoder.train()
decoder = decoder.to(device)
decoder.train()

# Define Loss
def maskNLLLoss(inp, target, mask):
	nTotal = mask.sum()
	# Compute -log of probability assigned by decoder to correct next word
	CrossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
	loss = CrossEntropy.masked_select(mask).mean()
	loss = loss.to(device)
	return loss, nTotal.item()


# ============================== Training loop  ============================== #

tb = SummaryWriter(comment='/' + xp_name + '_lr' + lr_str + '_bs' + str(batch_size) + '_ed' + str(emb_dim) + '_hs' + str(hidden_size))

for e in range(epoch, epochs):
	#print(f'Starting epoch {epoch}...')

	losses = []
	total_loss = 0
	n_totals = 0

	for batch in dataloader:
		idx_, inputs, _, targets, masks, targets_len = batch
		# print('input: ', inputs, '\n', 'target: ', targets, '\n', 'target_len', targets_len)
		# print(idx_.shape, inputs.shape, targets.shape, masks.shape, max_target_len.shape)
		# Get inquiries (inputs), replies (targets) and masks to proper shape for GRU: [seq_len, batch_size, emb_dim] (emb_dim is added by Encoder's embedder before passing seq through GRU)
		inputs, targets, masks = inputs.transpose(1, 0), targets.transpose(1, 0), masks.transpose(1, 0)
		# Get them to proper device
		inputs, targets, masks = inputs.to(device), targets.to(device), masks.to(device)

		encoder_optimizer.zero_grad()
		decoder_optimizer.zero_grad()

		# Initialize loss to zero
		batch_loss = 0

		# Forward pass through encoder
		encoder_outputs, encoder_hidden = encoder(inputs)

		# First input do decoder: SOS_token
		decoder_input = targets[0].unsqueeze(0)
		decoder_input = decoder_input.to(device)

		decoder_hidden = encoder_hidden[:decoder.n_layers]

		# Determine if teacher_forcing is used in this iteration
		use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

		#
		if use_teacher_forcing:
			for t in range(1, max(targets_len)):
				decoder_output, decoder_hidden = decoder(
					decoder_input, encoder_outputs, decoder_hidden
				)

				decoder_input = targets[t].view(1, -1)
				# print('decoder_output and shape: ', decoder_output, decoder_output.shape )
				# print('target: ', targets[t].view(-1, 1))
				# print('log input: ', torch.gather(decoder_output, 1, targets[t].view(-1, 1)).squeeze(1))
				# print('mask and product: ', masks[t], -torch.log(torch.gather(decoder_output, 1, targets[t].view(-1, 1)).squeeze(1))*masks[t] )

				mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], masks[t])
				batch_loss += mask_loss
				# print(batch_loss, nTotal)
				losses.append(mask_loss.item() * nTotal)
				n_totals += nTotal
		else:
			for t in range(1, max(targets_len)):
				decoder_output, decoder_hidden = decoder(
					decoder_input, encoder_outputs, decoder_hidden
				)
				_, topi = decoder_output.topk(1)
				decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
				decoder_input = decoder_input.to(device)

				mask_loss, nTotal = maskNLLLoss(decoder_output, targets[t], masks[t])
				batch_loss += mask_loss
				losses.append(mask_loss.item() * nTotal)
				n_totals += nTotal

		batch_loss.backward()

		_ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
		_ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

		encoder_optimizer.step()
		decoder_optimizer.step()
		# print(sum(losses), n_totals)
		total_loss = (sum(losses) / n_totals)
		encoder_scheduler.step(total_loss)
		decoder_scheduler.step(total_loss)

	tb.add_scalar('Total loss', total_loss, e)

	if e % 1 == 0:
		torch.save({'epoch': e, 'encoder': encoder.state_dict(), 'encoder_optim': encoder_optimizer.state_dict(),
					'decoder': decoder.state_dict(), 'decoder_optim': decoder_optimizer.state_dict(),
					'embedding': embedding.state_dict()}, os.path.join(folder_name, 'checkpoint.pth'))


tb.close()

torch.save({'epoch': e, 'encoder': encoder.state_dict(), 'encoder_optim': encoder_optimizer.state_dict(),
			'decoder': decoder.state_dict(), 'decoder_optim': decoder_optimizer.state_dict(),
			'embedding': embedding.state_dict()}, os.path.join(folder_name, 'checkpoint.pth'))
