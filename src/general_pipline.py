import logging
from load_dataset import Dataset
from feature_maping import Tokenize, Vocab
from url_phishing import UrlPhish

def main():

	logging.basicConfig(filename='phishing_log.app', filemode='a', level=logging.DEBUG, format='[%(asctime)s] - %(name)s - %(levelname)s - %(message)s')
	logger = logging.getLogger('phishing')

	token_char = Tokenize(lower=True, max_length=None, mode='char', ngram_size=1)
	token_word = Tokenize(lower=True, max_length=None, mode='word', ngram_size=1)
	print('tokenize ok')
	# field_tok = {'label':None, 'url_char':token_char.tokenize, 'url_word':token_word.tokenize}

	# train_data = Dataset('./train.csv', field_tok)

	# freq = Dataset.get_token_frequency(train_data)

	# sen_freq = freq['url_char'] + freq['url_word']

	# label_freq = freq['label']


	# sen_vocab = Vocab(sen_freq)
	# to_save(sen_vocab, './sen_vocab.pkl', 'sen_vocab', 'pickle')

	# print('Sent. Vocab.: ', sen_vocab)
	# print('Sent. type.: ', type(sen_vocab))

	# label_vocab = Vocab(label_freq, unk_token=None, pad_token=None)
	# to_save(label_vocab, './label_vocab.pkl', 'label_vocab', 'pickle')

	sen_vocab = Dataset.load_file_saved('data/sen_vocab.pkl', 'sen_vocab', 'pickle')

	label_vocab = Dataset.load_file_saved('data/label_vocab.pkl', 'label_vocab', 'pickle')
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


	# train = Dataset('./train.csv', tokenizers=tokenizer, numericalizers=numericalize)
	# to_save(train.data, './train_map.data', 'train map', 'data')

	# val = Dataset('./val.csv', tokenizers=tokenizer, numericalizers=numericalize)
	# to_save(val.data, './val_map.data', 'val map', 'data')
	
	# test = Dataset('./test.csv', tokenizers=tokenizer, numericalizers=numericalize)
	# to_save(test.data, './test_map.data', 'test map', 'data')

	train = Dataset.load_file_saved('data/train_map.data', 'train load', 'data')
	val = Dataset.load_file_saved('data/val_map.data', 'val load', 'data')
	test = Dataset.load_file_saved('data/test_map.data', 'test load', 'data')

	field_pad_index = {
						'url_char': sen_vocab[sen_vocab.pad_token],
						'url_word': sen_vocab[sen_vocab.pad_token], 
						'label':label_vocab[label_vocab.pad_token]
						}


	


	# logger.info('Setup data loaders')
	# train_loader = DataLoader(train, batch_size=batch, shuffle=True, collate_fn=collator.collate)
	# val_loader = DataLoader(val, batch_size=batch, shuffle=True, collate_fn=collator.collate)
	# test_loader = DataLoader(test, batch_size=batch, shuffle=True, collate_fn=collator.collate)

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
	batch_size = 900

	model = UrlPhish(input_dim,
					 embedding_dim, hidden_dim, 
					 n_lstm_layers, bidirectional,
					 n_fc_layers, output_dim,
					 dropout, pad_idx)

	kwargs = {'model':model,
			 'epochs':epochs,
			 'batch_size':batch_size,
			 'train_dataset':train,
			 # 'val_dataset':val,
			 'val_dataset':None,
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


main()
