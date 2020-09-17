import logging
from load_dataset import Dataset
from feature_maping import Tokenize, Vocab
from url_phishing import UrlPhish, UrlPhishCollator

from torch.utils.data import DataLoader
import torch
import argparse

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

	parser.add_argument('--phishing_log', '-l',
						dest='log',
						required=True,
						help='log file name')

	parser.add_argument('--save_files', '-s',
						dest='use_saved',
						required=True,
						help='if save (True) or not (False) the generated files')

	parser.add_argument('--training_set', '-t',
						dest='training_set',
						required=False,
						help='Training set path')

	parser.add_argument('--testing_set', '-v',
						dest='validation_set',
						required=False,
						help='Validation set path')

	parser.add_argument('--model_path_name', '-m',
						dest='model_path_name',
						required=True,
						help='model path name')


	return parser.parse_args()

def main():

	args = command_line_parsing()

	logging.basicConfig(filename=args.log, filemode='a', level=logging.DEBUG, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
	logger = logging.getLogger('phishing')

	logger.info('Tokenizing...')

	token_char = Tokenize(lower=True, max_length=None, mode='char', ngram_size=1)
	# token_word = Tokenize(lower=True, max_length=None, mode='word', ngram_size=1)
	

	# -----------------------------------------------------------------------------------
	field_tok = {'label':None, 
				 'url_char':token_char.tokenize
	 			 # , 'url_word':token_word.tokenize
	 			 }

	if (args.use_saved.lower() == 'false'):

		# train_data = Dataset('./train.csv' or new_train.csv, field_tok)
		train_data = Dataset(args.training_set, field_tok)
		freq = Dataset.get_token_frequency(train_data)

		# sen_freq = freq['url_char'] + freq['url_word']
		sen_freq = freq['url_char']
		
		label_freq = freq['label']

		logger.info('Building vocabulary...')

		vocab_sen = Vocab(sen_freq)
		vocab_label = Vocab(label_freq, unk_token=None, pad_token=None)

		try:

			logger.info('Saving vocab_sen and vocab_label')

			Dataset.to_save(vocab_sen, './vocab_sen.pkl', 'vocab_sen', 'pickle')
			Dataset.to_save(vocab_label, './vocab_label.pkl', 'vocab_label', 'pickle')

		except Exception as e:

			logger.error('Error on saving vocab_sen.pkl or vocab_label.pkl')
			logger.error(e)
			
			exit()

	else: 

		try:

			logger.info('Loading vocab_sen and vocab_label')

			vocab_sen = Dataset.load_file_saved('./vocab_sen.pkl', './vocab_sen.pkl', 'pickle')
			vocab_label = Dataset.load_file_saved('./vocab_label.pkl', './vocab_label.pkl', 'pickle')

		except Exception as e:

			logger.error('Error on saving vocab_sen.pkl or vocab_label.pkl')
			logger.error(e)

			exit()

	# -----------------------------------------------------------------------------------

	tokenizer = {
				'url_char': token_char.tokenize,
				# 'url_word': token_word.tokenize, 
				'label':None
				}

	numericalize = {
					'url_char': vocab_sen.numericalize,
					# 'url_word': vocab_sen.numericalize, 
					'label':vocab_label.numericalize
					}


	if (args.use_saved.lower() == 'false'):

		train = Dataset(args.training_set, tokenizers=tokenizer, numericalizers=numericalize)
		val = Dataset(args.validation_set, tokenizers=tokenizer, numericalizers=numericalize)

		try:

			logger.info('Saiving training and validation mapping')

			Dataset.to_save(train.data, './map_train.data', 'train map', 'data')
			Dataset.to_save(val.data, './map_val.data', 'val map', 'data')

		except Exception as e:

			logger.error('Error on saving mapping files')
			logger.error(e)

	else:

		try:

			logger.info('Loading training and validation mapping')

			train = Dataset.load_file_saved('./map_train.data', 'train map', 'data')
			val = Dataset.load_file_saved('./map_val.data', 'val map', 'data')

		except Exception as e:

			logger.error('Error on loading mapping files')
			logger.error(e)


	field_pad_index = {
						'url_char': vocab_sen[vocab_sen.pad_token],
						# 'url_word': vocab_sen[vocab_sen.pad_token], 
						'label':vocab_label[vocab_label.pad_token]
						}


	input_dim = len(vocab_sen)
	# input_dim_word = len(word_vocab)

	embedding_dim = 10 # 100
	hidden_dim = 300
	n_lstm_layers = 1
	bidirectional = True
	n_fc_layers = 3
	output_dim = len(vocab_label)
	dropout = 0.2
	pad_idx = vocab_sen[vocab_sen.pad_token]
	epochs = 100
	batch_size = 32 #128

	model = UrlPhish(input_dim,
					 embedding_dim, hidden_dim, 
					 n_lstm_layers, bidirectional,
					 n_fc_layers, output_dim,
					 dropout, pad_idx, args.log)

	kwargs = {'model':model,
			 'epochs':epochs,
			 'batch_size':batch_size,
			 'train_dataset':train,
			 'val_dataset':val,
			 #'val_dataset':None,
			 'field_pad_index':field_pad_index,
			 'log' : args.log,
			 'model_path_name':args.model_path_name
			 }

	train_final_acc, train_final_loss, val_final_acc, val_final_acc = UrlPhish.train_model(**kwargs)

	url_char = 'https://ifood.mobi/sms'
	#url_word = 'https://ifood.mobi/sms'

	kwargs = {
	'model' : model,
	'url_char' : url_char,
	# 'url_word' : url_word,
	'char_tokenizer': token_char.tokenize,
	#'word_tokenizer': token_word.tokenize,
	'sentence_numericalizer': vocab_sen.numericalize,
	'vocab_label': vocab_label,
	'log' : args.log
	}
	
	answer = UrlPhish.infer(**kwargs)


	print('Inference for ', url_char, ' was: ', answer)

	logger.debug('Inference for %s is %s', url_char, answer)

	


main()
