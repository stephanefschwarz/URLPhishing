import logging
from load_dataset import Dataset
from feature_maping import Tokenize, Vocab
from url_phishing import UrlPhish, UrlPhishCollator

from torch.utils.data import DataLoader
import torch

def main():

	logging.basicConfig(filename='phishing_log.app', filemode='a', level=logging.DEBUG, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
	logger = logging.getLogger('phishing')

	token_char = Tokenize(lower=True, max_length=None, mode='char', ngram_size=1)
	token_word = Tokenize(lower=True, max_length=None, mode='word', ngram_size=1)
	print('tokenize ok')
	logger.info('Tokenize ok')
	# -----------------------------------------------------------------------------------
	field_tok = {'label':None, 'url_char':token_char.tokenize, 'url_word':token_word.tokenize}

	#train_data = Dataset('./train.csv', field_tok)

	#logger.info('Train data')
	#print("TRAIN DATA \n")

	#freq = Dataset.get_token_frequency(train_data)
	#logger.info('Frequency')
	#sen_freq = freq['url_char'] + freq['url_word']

	#label_freq = freq['label']

	#logger.info('Label')

	#sen_vocab = Vocab(sen_freq)

	logger.info('Loading sen_vocab')
	try:
		#Dataset.to_save(sen_vocab, './sen_vocab.pkl', 'sen_vocab', 'pickle')
		sen_vocab = Dataset.load_file_saved('./sen_vocab.pkl', './sen_vocab.pkl', 'pickle')

	except Exception as e:

		logger.info('Error on saving sen_vocab.pkl')
		logger.info(e)
		exit()

	#print('Sent. Vocab.: ', sen_vocab)
	#print('Sent. type.: ', type(sen_vocab))

	#label_vocab = Vocab(label_freq, unk_token=None, pad_token=None)
	try:
		logger.info('Loading label vocabulary')
		#Dataset.to_save(label_vocab, './label_vocab.pkl', 'label_vocab', 'pickle')
		label_vocab = Dataset.load_file_saved('./label_vocab.pkl', './label_vocab.pkl', 'pickle')

	except:
		logger.info('Error on saving label_vocab.pkl')
		exit()

	# -----------------------------------------------------------------------------------

	#sen_vocab = Dataset.load_file_saved('./sen_vocab.pkl', 'sen_vocab', 'pickle')
	#print('Sent. Vocab.: ', sen_vocab)

	#label_vocab = Dataset.load_file_saved('data/label_vocab.pkl', 'label_vocab', 'pickle')
	logging.info('Loaded files.')
	tokenizer = {
				'url_char': token_char.tokenize,
				'url_word': token_word.tokenize, 
				'label':None
				}

	numericalize = {
					'url_char': sen_vocab.numericalize,
					'url_word': sen_vocab.numericalize, 
					'label':label_vocab.numericalize
					}

	logger.info('Tokenizing train set')
	#train = Dataset('./train.csv', tokenizers=tokenizer, numericalizers=numericalize)
	#logger.info('Saiving tekenized train set')
	#Dataset.to_save(train.data, './train_map.data', 'train map', 'data')
	train = Dataset.load_file_saved('./train_map.data', 'train map', 'data')

	logger.info('Tokenizing val set')
	val = Dataset.load_file_saved('./val_map.data', 'val map', 'data')


	#val = Dataset('./val.csv', tokenizers=tokenizer, numericalizers=numericalize)
	#logger.info('Saving tokenized val set')
	#Dataset.to_save(val.data, './val_map.data', 'val map', 'data')
	
	#test = Dataset('./test.csv', tokenizers=tokenizer, numericalizers=numericalize)
	#Dataset.to_save(test.data, './test_map.data', 'test map', 'data')

	#train = Dataset.load_file_saved('data/train_map.data', 'train load', 'data')
	#val = Dataset.load_file_saved('data/val_map.data', 'val load', 'data')
	#test = Dataset.load_file_saved('data/test_map.data', 'test load', 'data')

	field_pad_index = {
						'url_char': sen_vocab[sen_vocab.pad_token],
						'url_word': sen_vocab[sen_vocab.pad_token], 
						'label':label_vocab[label_vocab.pad_token]
						}


	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	logging.debug('Device: %s', device)

	print('\nWe will use the ', device, 'device. \n')

	collator = UrlPhishCollator(field_pad_index, device)

	batch = 128
	logger.info('Setup data loaders')
	train_loader = DataLoader(train, batch_size=batch, shuffle=True, collate_fn=collator.collate)
	val_loader = DataLoader(val, batch_size=batch, shuffle=True, collate_fn=collator.collate)
	#test_loader = DataLoader(test, batch_size=batch, shuffle=True, collate_fn=collator.collate)

	input_dim = len(sen_vocab)
	# input_dim_word = len(word_vocab)


	embedding_dim = 100
	hidden_dim = 300
	n_lstm_layers = 1
	bidirectional = True
	n_fc_layers = 3
	output_dim = len(label_vocab)
	dropout = 0.2
	pad_idx = sen_vocab[sen_vocab.pad_token]
	epochs = 3
	batch_size = 128

	collete = collator

	model = UrlPhish(input_dim,
					 embedding_dim, hidden_dim, 
					 n_lstm_layers, bidirectional,
					 n_fc_layers, output_dim,
					 dropout, pad_idx)

	kwargs = {'model':model,
			 'epochs':epochs,
			 'batch_size':batch_size,
			 'train_dataset':train,
			 'val_dataset':val,
			 #'val_dataset':None,
			 'field_pad_index':field_pad_index
			 }

	acc, loss = UrlPhish.train_model(**kwargs)

	url_char = 'https://w1.smsaapf.com/'
	url_word = 'https://w1.smsaapf.com/'

	kwargs = {
	'model' : model,
	'url_char' : url_char,
	'url_word' : url_word,
	'char_tokenizer': char_tokenizer,
	'word_tokenizer': word_tokenizer,
	'sentence_numericalizer': sentence_numericalizer,
	'label_vocab': label_vocab
	}
	
	answer = UrlPhish.infer(**kwargs)


	print('Inference for ', url_word, ' was: ', answer)

	logger.debug('Inference for %s is %s', url_word, answer)

	print('Saving model...')

	torch.save(model.state_dict(), './modeloEsse')

	


main()
