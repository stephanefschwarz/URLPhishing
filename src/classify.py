import logging
from load_dataset import Dataset
from feature_maping import Tokenize, Vocab
from url_phishing import UrlPhish, UrlPhishCollator
import argparse
import os.path

from torch.utils.data import DataLoader
import torch

def command_line_parsing():
	"""Parse command lines
		Parameters
		----------
		model_path : str
			path to the train dataset

		model : URLPhish
			path to the train dataset
		Returns
		-------
		parser
			The arguments from command line
	"""
	parser = argparse.ArgumentParser(description = __doc__)

	parser.add_argument('--url', '-u',
						dest='url',
						required=True,
						help='url')

	parser.add_argument('--model-path', '-i',
						dest='model_path',
						required=True,
						help='File path for the pre trained model.')

	parser.add_argument('--sen_vocab', '-s',
						dest='sen_vocab_path',
						required=True,
						help='Path for the sen_vocab.pkl file.')
	
	parser.add_argument('--label_vocab', '-l',
						dest='label_vocab_path',
						required=True,
						help='Path for the label_vocab.pkl file.')
	
	parser.add_argument('--report_path', '-r',
						dest='report_path',
						required=True,
						help='Path for the report file .csv')

	return parser.parse_args()

def main():

	args = command_line_parsing()

	url_char = url_word = args.url
	#print(url_char, url_word)
	token_char = Tokenize(lower=True, max_length=None, mode='char', ngram_size=1)
	#token_word = Tokenize(lower=True, max_length=None, mode='word', ngram_size=1)

	sen_vocab = Dataset.load_file_saved(args.sen_vocab_path, './sen_vocab.pkl', 'pickle')

	label_vocab = Dataset.load_file_saved(args.label_vocab_path, './label_vocab.pkl', 'pickle')

	input_dim = len(sen_vocab)
	embedding_dim = 10 #100
	hidden_dim = 300
	n_lstm_layers = 1
	bidirectional = True
	n_fc_layers = 3
	output_dim = len(label_vocab)
	dropout = 0.2
	pad_idx = sen_vocab[sen_vocab.pad_token]

	epochs = 3
	batch_size = 128

	model = UrlPhish(input_dim,
					 embedding_dim, hidden_dim, 
					 n_lstm_layers, bidirectional,
					 n_fc_layers, output_dim,
					 dropout, pad_idx, args.report_path)

	model_ = UrlPhish.load_model(model, args.model_path)
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model_.to(device)

	kwargs = {
		'model' : model_,
		'url_char' : url_char,
		#'url_word' : url_word,
		'char_tokenizer': token_char.tokenize,
		#'word_tokenizer': token_word.tokenize,
		'sentence_numericalizer': sen_vocab.numericalize,
		'vocab_label': label_vocab,
		'log' : args.report_path
		}


	answer = UrlPhish.infer(**kwargs)

	if os.path.exists(args.report_path) == False:

		file = open(args.report_path, 'a')

		file.write('url,predicted_label\n')

	else:

		file = open(args.report_path, 'a')

	file.write('{},{}\n'.format(args.url, answer))
	file.close()



	print('The ', args.url, ' URL is ', answer)


main()
