import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import pandas as pd
from tqdm import tqdm

from load_dataset import Dataset

import logging
from logging.config import fileConfig

logging.basicConfig(filename='phishing_log.app', filemode='a', level=logging.DEBUG, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('phishing')


class UrlPhish(nn.Module):

	def __init__(self, vocab_size, embedding_dim,
				 hidden_dim, n_lstm_layers, bidirectional,
				 n_fc_layers, output_dim,
				 dropout, pad_idx
				 ):

		super().__init__()

		self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, 
																	padding_idx=pad_idx)
		
		self.translation = nn.Linear(in_features=embedding_dim, out_features=hidden_dim)

		self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim,
												num_layers=n_lstm_layers, bidirectional=bidirectional,
												dropout=dropout if n_lstm_layers > 1 else 0)
		
		fc_dim = hidden_dim * 2 if bidirectional else hidden_dim

		fcs = [nn.Linear(fc_dim * 2, fc_dim * 2) for _ in range(n_fc_layers)]

		self.fcs = nn.ModuleList(modules=fcs)

		self.fc_out = nn.Linear(in_features=fc_dim * 2, out_features=output_dim)

		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, char_url, word_url):

		char_seq_len, batch_size = char_url.shape
		word_seq_len, _ = word_url.shape

		embedded_char = self.embedding(char_url)
		embedded_word = self.embedding(word_url)

		translated_char = F.relu(self.translation(embedded_char))
		translated_word = F.relu(self.translation(embedded_word))

		outputs_char, (hidden_char, cell_char) = self.lstm(translated_char)
		outputs_word, (hidden_word, cell_word) = self.lstm(translated_word)

		if self.lstm.bidirectional:

			hidden_char = torch.cat((hidden_char[-1], hidden_char[-2]), dim=-1)
			hidden_word = torch.cat((hidden_word[-1], hidden_word[-2]), dim=-1)

		else:

			hidden_char = hidden_char[-1]
			hidden_word = hidden_word[-1]

		hidden = torch.cat((hidden_char, hidden_word), dim=1)

		for fc in self.fcs:
			hidden = fc(hidden)
			hidden = F.relu(hidden)
			hidden = self.dropout(hidden)

		prediction = self.fc_out(hidden)

		return prediction


# ============================================================================================
# ============================================================================================
	
	@staticmethod
	def train_model(**kwargs):

		# ======================================================

		def training(model, iterator, optimizer, criterion):

			epoch_loss = 0
			epoch_acc = 0

			model.train()

			for labels, char, word in tqdm(iterator):

				optimizer.zero_grad()

				predictions = model(char, word)

				loss = criterion(predictions, labels)

				acc = categorical_accuracy(predictions, labels)
				loss.backward()

				optimizer.step()

				epoch_acc = epoch_acc + acc.item()
				epoch_loss = epoch_loss + loss.item()


			return epoch_loss / len(iterator), epoch_acc / len(iterator)		

		# ======================================================

		def evaluate(model, iterator, criterion):

			epoch_loss = 0
			epoch_acc = 0

			model.eval()

			with torch.no_grad():

				for labels, prems, hypos in tqdm(iterator):

					predictions = model(prems, hypos)

					loss = criterion(predictions, labels)

					acc = categorical_accuracy(predictions, labels)

					epoch_loss += loss.item()
					epoch_acc += acc.item()

			return epoch_loss / len(iterator), epoch_acc / len(iterator)

		# ======================================================

		def categorical_accuracy(preds, labels):

			max_preds = preds.argmax(dim=-1) 

			correct = max_preds.eq(labels)

			return correct.sum() / torch.FloatTensor([labels.shape[0]])

		# ======================================================

		train_final_acc = []
		train_final_loss = []

		val_final_acc = []
		val_final_loss = []

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		model = kwargs['model']
		epochs = kwargs['epochs']
		batch_size = kwargs['batch_size']
		dataset = kwargs['train_dataset']
		val_dataset = kwargs['val_dataset']
		field_pad_index = kwargs['field_pad_index']
		val_iterator = None

		collator = UrlPhishCollator(field_pad_index, device)

		iterator = DataLoader(dataset, batch_size=batch_size,shuffle=True, collate_fn= collator.collate)
			

		if (val_dataset != None):

			val_iterator = DataLoader(val_dataset, batch_size=batch_size,shuffle=True, collate_fn= collator.collate)


		optimizer = optim.Adam(model.parameters())

		criterion = nn.CrossEntropyLoss()

		model = model.to(device)

		criterion = criterion.to(device)

		for epoch in range(epochs):

			train_loss, train_acc = training(model, iterator, optimizer, criterion)

			logger.debug('Train Epoch %s \t --> \t acc: %s \t\t loss: %s', epoch, train_acc, train_loss)

			train_final_loss.append(train_loss)
			train_final_acc.append(train_acc)

			if(val_iterator != None):
				
				val_loss, val_acc = evaluate(model, val_iterator, criterion)

				val_final_acc.append(val_loss)
				val_final_loss.append(val_acc)

				logger.debug('Test Epoch %s \t --> \t acc: %s \t\t loss: %s', epoch, val_acc, val_loss)


		checkpoint = {
					'epoch': epochs,
					'model_state_dict': model.state_dict(),
					'optimizer_state_dict': optimizer.state_dict()
					# 'loss': loss
					}

		torch.save(checkpoint,'./model.pth.tar')


		return (train_final_acc, train_final_loss, 
			    val_final_acc, val_final_acc)


	@staticmethod
	def infer(**kwargs):

		model = kwargs['model']
		url_char = kwargs['url_char']
		url_word = kwargs['url_word']
		char_tokenizer = kwargs['char_tokenizer']
		word_tokenizer = kwargs['word_tokenizer']
		sentence_numericalizer = kwargs['sentence_numericalizer']
		label_vocab = kwargs['label_vocab']

		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

		model.eval()

		if isinstance(url_char, str):
			url_char = char_tokenizer(url_char)

		if isinstance(url_word, str):
			url_word = word_tokenizer(url_word)

		url_char = sentence_numericalizer(url_char)
		url_word = sentence_numericalizer(url_word)

		url_char = torch.LongTensor(url_char).unsqueeze(1).to(device)
		url_word = torch.LongTensor(url_word).unsqueeze(1).to(device)

		prediction = model(url_char, url_word)

		prediction = prediction.argmax(dim=-1).item()

		return label_vocab.int_to_str[prediction]


	@staticmethod
	def load_model(model, optimizer, model_path):

		checkpoint = torch.load(model_path)

		model.load_state_dict(checkpoint['model_state_dict'])

		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

		epoch = checkpoint['epoch']

		loss = checkpoint['loss']

		return (model, optimizer, epoch, loss)

# ============================================================================================ #
# ============================================================================================ #

class UrlPhishCollator:

	def __init__(self, field_pad_inxs, device, batch_first=False):

		self.field_pad_inxs = field_pad_inxs
		self.device = device
		self.batch_first = batch_first

	def collate(self, batch):

		labels = []
		url_char = []
		url_word = []

		for sample in batch:

			labels.append(sample['label'])
			url_char.append(torch.LongTensor(sample['url_char']))
			url_word.append(torch.LongTensor(sample['url_word']))

		url_char = pad_sequence(url_char, padding_value=self.field_pad_inxs['url_char'],
															batch_first=self.batch_first).to(self.device)

		url_word = pad_sequence(url_word, padding_value=self.field_pad_inxs['url_word'],
															batch_first=self.batch_first).to(self.device)

		labels = torch.LongTensor(labels).to(self.device)

		return labels, url_char, url_word

