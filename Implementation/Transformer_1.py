import os, re
import pickle
import math, copy
import unicodedata
import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.normalization import LayerNorm
from tensorboardX import SummaryWriter
import dataset_vf1 as dataset

# Working Directory, Experiment Name
wd = os.getcwd()
xp_name = 'Transformer_1_1'

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


# =================================== Utils ==================================== #

# Clone modules
def get_clones(module, N):
	return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


# Masks
def nopeak_mask(size):
	np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
	np_mask = torch.from_numpy(np_mask) == 0
	return np_mask


def create_masks(src, trg, device):
	src_mask = (src != 0).unsqueeze(-2)
	# src_mask = src_mask.to(device)
	if trg is not None:
		trg_mask = (trg != 0).unsqueeze(-2)
		trg_mask = trg_mask.to(device)
		size = trg.size(1)
		np_mask = nopeak_mask(size)
		np_mask = np_mask.to(device)
		trg_mask = trg_mask & np_mask
	else:
		trg_mask = None
	return src_mask, trg_mask


# ============================ Transformer Modules ============================= #

# Word position encoding
class PositionalEncoder(nn.Module):
	def __init__(self, emb_dim, max_seq_len=200, dropout=0.1):
		super().__init__()
		self.emb_dim = emb_dim
		self.wpe = nn.Embedding(max_seq_len, emb_dim)  # word position encoding
		self.dropout = nn.Dropout(dropout)

	def forward(self, x):
		pos_ids = torch.arange(0, x.size(1), device=device).unsqueeze(0)
		xpe = self.wpe(pos_ids)
		return xpe


# Attention
class MultiHeadAttention(nn.Module):
	def __init__(self, n_heads, emb_dim, dim_k=None, dropout=0.1):
		super().__init__()
		self.emb_dim = emb_dim
		self.dim_k = dim_k if dim_k else emb_dim // n_heads
		self.n_heads = n_heads
		self.q_linear = nn.Linear(emb_dim, self.dim_k * n_heads)
		self.k_linear = nn.Linear(emb_dim, self.dim_k * n_heads)
		self.v_linear = nn.Linear(emb_dim, self.dim_k * n_heads)

		self.dropout = nn.Dropout(dropout)
		self.out = nn.Linear(self.dim_k * n_heads, emb_dim)

	def attention(self, q, k, v, dim_k, mask=None, dropout=None):
		k = k.transpose(-2, -1)  # prepare k's shape for matmul
		scores = torch.matmul(q, k) / math.sqrt(dim_k)  # [batch_size, n_heads, seq_len, seq_len]
		if mask is not None:
			mask = mask.unsqueeze(1)
			scores = scores.masked_fill(mask == 0, -1e9)
		softscores = F.softmax(scores, dim=-1)
		if dropout is not None: softscores = dropout(softscores)
		output = torch.matmul(softscores, v)  # [batch_size, n_heads, seq_len, dim_k]
		return output, scores

	def forward(self, q, k, v, mask=None):
		batch_size = q.size(0)
		q = self.q_linear(q)  # [batch_size, seq_len, emd_dim]
		k = self.k_linear(k)  # [batch_size, seq_len, emd_dim]
		v = self.v_linear(v)  # [batch_size, seq_len, emd_dim]

		k = k.view(batch_size, -1, self.n_heads, self.dim_k)  # [batch_size, seq_len, n_heads, dim_k]
		q = q.view(batch_size, -1, self.n_heads, self.dim_k)  # [batch_size, seq_len, n_heads, dim_k]
		v = v.view(batch_size, -1, self.n_heads, self.dim_k)  # [batch_size, seq_len, n_heads, dim_k]

		k = k.transpose(1, 2)  # [batch_size, n_heads, seq_len, dim_k]
		q = q.transpose(1, 2)  # [batch_size, n_heads, seq_len, dim_k]
		v = v.transpose(1, 2)  # [batch_size, n_heads, seq_len, dim_k]

		attn, scores = self.attention(q, k, v, self.dim_k, mask, self.dropout)
		concat = attn.transpose(1, 2).contiguous().view(batch_size, -1,
														self.dim_k * self.n_heads)  # [batch_size, seq_len, emb_dim]
		output = self.out(concat)
		return output, scores


# Feed Forward
class FeedForward(nn.Module):
	def __init__(self, emb_dim, ffwd_dim=2048, dropout=0.1):
		super().__init__()
		self.linear1 = nn.Linear(emb_dim, ffwd_dim)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(ffwd_dim, emb_dim)

	def forward(self, x):
		x = self.dropout(F.leaky_relu(self.linear1(x)))  # clarify why Leaky  (GPT2 uses gelu)
		x = self.linear2(x)
		return x


# Encoder
class EncoderLayer(nn.Module):
	def __init__(self, emb_dim, heads, dropout=0.1):
		super().__init__()
		self.ln1 = LayerNorm(emb_dim)
		self.dropout1 = nn.Dropout(dropout)
		self.attn = MultiHeadAttention(n_heads=heads, emb_dim=emb_dim, dropout=dropout)
		self.ln2 = LayerNorm(emb_dim)
		self.ffwd = FeedForward(emb_dim, dropout=dropout)
		self.dropout2 = nn.Dropout(dropout)

	def forward(self, vector_sequence, mask):
		x = self.ln1(vector_sequence)
		x_attn, x_scores = self.attn(x, x, x,
									 mask)  # [batch_size, seq_len, emb_dim], [batch_size, n_heads, seq_len, seq_len]
		vector_sequence = vector_sequence + self.dropout1(x_attn)
		x = self.ln2(vector_sequence)
		vector_sequence = vector_sequence + self.dropout2(self.ffwd(x))  # ¿CLARIFY USE OF FEEDFORWARD?
		return vector_sequence  # [batch_size, seq_len, emb_dim]


class Encoder(nn.Module):
	def __init__(self, emb_dim, embedding, n_layers, heads, dropout):
		super().__init__()
		self.n_layers = n_layers
		self.wte = embedding  # word token embedding
		self.wpe = PositionalEncoder(emb_dim)  # word position encoding
		self.layers = get_clones(EncoderLayer(emb_dim, heads, dropout=dropout), n_layers)
		self.ln = LayerNorm(emb_dim)

	def forward(self, source_seq, source_mask):
		x = self.wte(source_seq)  # [batch_size, seq_len, emb_dim]
		x = x + self.wpe(source_seq)  # [batch_size, seq_len, emb_dim]
		for i in range(self.n_layers):
			x = self.layers[i](x, source_mask)
		x = self.ln(x)
		return x


# Decoder
class DecoderLayer(nn.Module):
	def __init__(self, emb_dim, heads, dropout=0.1):
		super().__init__()
		self.ln1 = LayerNorm(emb_dim)
		self.ln2 = LayerNorm(emb_dim)
		self.ln3 = LayerNorm(emb_dim)

		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)

		self.attn1 = MultiHeadAttention(heads, emb_dim, dropout=dropout)
		self.attn2 = MultiHeadAttention(heads, emb_dim, dropout=dropout)
		self.ffwd = FeedForward(emb_dim, dropout=dropout)

	def forward(self, de_out, de_mask, en_out, en_mask):
		de_nrm = self.ln1(de_out)
		# Self Attention
		self_attn, self_scores = self.attn1(de_nrm, de_nrm, de_nrm, de_mask)
		de_out = de_out + self.dropout1(self_attn)
		de_nrm = self.ln2(de_out)
		# Encoder-Decoder Attention
		en_attn, en_scores = self.attn2(de_nrm, en_out, en_out, en_mask)
		de_out = de_out + self.dropout2(en_attn)
		de_nrm = self.ln3(de_out)
		de_out = de_out + self.dropout3(self.ffwd(de_nrm))
		return de_out


class Decoder(nn.Module):
	def __init__(self, emb_dim, embedding, n_layers, heads, dropout):
		super().__init__()
		self.n_layers = n_layers
		self.wte = embedding
		self.wpe = PositionalEncoder(emb_dim)
		self.layers = get_clones(DecoderLayer(emb_dim, heads, dropout), n_layers)
		self.ln = LayerNorm(emb_dim)

	def forward(self, de_toks, de_mask, en_vecs, en_mask):
		x = self.wte(de_toks)
		x = x + self.wpe(de_toks)
		for i in range(self.n_layers):
			x = self.layers[i](x, de_mask, en_vecs, en_mask)
		return self.ln(x)


# Assemble Transformer elements
class Transformer(nn.Module):
	def __init__(self, vocab_size, emb_dim, n_layers, heads, dropout):
		super().__init__()
		self.encoder = Encoder(vocab_size, emb_dim, n_layers, heads, dropout)
		self.decoder = Decoder(vocab_size, emb_dim, n_layers, heads, dropout)
		self.out = nn.Linear(emb_dim, vocab_size)

	def forward(self, src_seq, trg_seq, src_mask, trg_mask):
		e_output = self.encoder(src_seq, src_mask)
		d_output = self.decoder(trg_seq, trg_mask, e_output, src_mask)
		output = self.out(d_output)
		return output


# ======================== Define and instantiate model ======================== #
epoch = 0

# DataLoader
ds = dataset.dataset(voc, pairs)
bs = 1
dataloader = data.DataLoader(ds, batch_size=bs)
#print(len(dataloader))

# Construct model, optimizer and lr_scheduler
emb_dim, n_layers, heads = 30, 3, 8
embedding = nn.Embedding(voc.num_words, emb_dim)
model = Transformer(vocab_size=voc.num_words, emb_dim=emb_dim, n_layers=n_layers, heads=heads, dropout=0.01)
model.to(device)
lr = 0.0001
lr_str = '0001'
optimizer = optim.Adam(model.parameters(), lr=lr)

# Continue experiment? Load model.state_dict() and optimizer.state_dict()
folder_name = os.path.join(wd, xp_name)

if os.path.exists(os.path.join(folder_name, 'checkpoint.pth')):
	try:
		checkpoint = torch.load(os.path.join(folder_name, 'checkpoint.pth'))
	except:
		raise FileNotFoundError("Unable to load checkpoint")
	epoch = checkpoint['epoch'] + 1
	optimizer.load_state_dict(checkpoint['optimizer'])
	try:
		params = torch.load(os.path.join(folder_name, xp_name + '_lr' + lr_str + '_bs' + str(bs) + '_ed' + str(emb_dim)
									 + '_nl' + str(n_layers) + '_h' + str(heads) + '_epoch' + str(epoch - 1) + '.pth'))
	except:
		raise FileNotFoundError("Unable to load pretrained parameters")
	model.load_state_dict(params)
	try:
		pre_embedding = torch.load(os.path.join(folder_name, 'embedding_epoch' + str(epoch - 1) + '.pth'))
	except:
		raise FileNotFoundError("Unable to load pretrained embedding")
	embedding.load_state_dict(pre_embedding)
else:
	os.mkdir(folder_name)

scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
# loss = nn.CrossEntropyLoss()

# Training loop
epochs = epoch + 3000

tb = SummaryWriter(comment='/' + xp_name + '_lr' + str(lr) + '_bs' + str(bs) + '_ed' + str(emb_dim)
						   + '_nl' + str(n_layers) + '_h' + str(heads))

# ============================== Training loop  ============================== #

for e in range(epoch, epochs):
	total_loss = 0
	for batch in dataloader:
		optimizer.zero_grad()
		# Get inputs and targets and get them into proper device
		_, inputs, _, targets, _, _ = batch
		inputs, targets = inputs.to(device), targets.to(device)
		# Get masks
		input_mask, target_mask = create_masks(inputs, targets, device)
		# print(inputs.device, targets.device, input_mask.device, target_mask.device)
		# print(next(model.parameters()).is_cuda)
		# Get predictions
		preds = model(inputs, targets, input_mask, target_mask)
		# Compute loss, perform gradient descent and update params
		ys = targets[:, :].contiguous().view(-1)
		batch_loss = F.cross_entropy(preds.view(-1, preds.size(-1)), ys)
		batch_loss.backward()
		optimizer.step()

		total_loss += batch_loss.item()

	epoch_loss = total_loss / (len(dataloader))
	scheduler.step(epoch_loss)

	tb.add_scalar('Total Loss', total_loss, e)
	tb.add_scalar('Epoch Loss', epoch_loss, e)

	if e % 1 == 0:
		torch.save({'epoch': e, 'optimizer': optimizer.state_dict()},
				   os.path.join(folder_name, 'checkpoint.pth'))
		torch.save(model.state_dict(),
				   os.path.join(folder_name, xp_name + '_lr' + lr_str + '_bs' + str(bs) + '_ed' + str(emb_dim)
								+ '_nl' + str(n_layers) + '_h' + str(heads) + '_epoch' + str(e) + '.pth'))
		torch.save(embedding.state_dict(), os.path.join(folder_name, 'embedding_epoch' + str(e) + '.pth'))

tb.close()

torch.save({'epoch': e, 'optimizer': optimizer.state_dict()},
		   os.path.join(folder_name, 'checkpoint.pth'))
torch.save(model.state_dict(),
		   os.path.join(folder_name, xp_name + '_lr' + lr_str + '_bs' + str(bs) + '_ed' + str(emb_dim)
						+ '_nl' + str(n_layers) + '_h' + str(heads) + '_epoch' + str(e) + '.pth'))
torch.save(embedding.state_dict(), os.path.join(folder_name, 'embedding_epoch' + str(e) + '.pth'))
