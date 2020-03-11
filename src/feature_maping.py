import re
import logging
from logging.config import fileConfig

class Tokenize:

	def __init__(self, lower=True, max_length=None, mode='word', ngram_size=1):

		self.lower = lower
		self.max_length = max_length
		self.mode = mode
		self.ngram_size = ngram_size

	def tokenize(self, url):

		if (self.mode.lower() == 'char'):

			tokens =  self.__generate_char_grams_from_URL(url)

		if (self.mode.lower() == 'word'):

		 tokens = self.__generate_word_grams_from_URL(url)

		if (self.max_length is not None):
			tokens = tokens[:self.max_length]
		
		if (self.lower):

			low_tokens = [token.lower() for token in tokens]

		return low_tokens


	def __generate_char_grams_from_URL(self, url):

		url_ngrams = []

		threshold = self.ngram_size if ((len(url) % 2) == 0) else (self.ngram_size - 1)

		for i in range(0, len(url)-threshold):

			ngrams = url[i:i+self.ngram_size]

			url_ngrams.append(ngrams)

		return url_ngrams

	def __generate_word_grams_from_URL(self, url):

		url_to_words = None

		url_to_words = (re.split(r'[^A-Za-z]+', url))

		if (self.ngram_size == 1):

			return url_to_words

		word_grams = []

		threshold = self.ngram_size if ((len(url_to_words) % 2) == 0) else (self.ngram_size - 1)
		
		for i in range(len(url_to_words) - threshold):

			string = ' '.join(url_to_words[i:i+self.ngram_size])

			word_grams.append(string)

		return word_grams

# ============================================================================================ #
# ============================================================================================ #

class Vocab:
	def __init__(self, frequencies, max_size=None, min_freq=1, 
							 pad_token='<PAD>', unk_token='<UNK>',
							 special_toks=[]):
		
		self.max_size = max_size
		self.min_freq = min_freq
		self.pad_token = pad_token
		self.unk_token = unk_token
		self.special_toks = special_toks

		self.int_to_str, self.str_to_int = self.build_vocab(frequencies)

	def build_vocab(self, frequencies):

		int_to_str = list()
		str_to_int = dict()

		if self.pad_token is not None:
			int_to_str.append(self.pad_token)

		if self.unk_token is not None:
			int_to_str.append(self.unk_token)

		for token in self.special_toks:
			int_to_str.append(token)
		
		for token, count in frequencies.most_common(self.max_size):

			if token in int_to_str:
				continue

			if count < self.min_freq:
				break

			else:
				int_to_str.append(token)

		str_to_int.update({token:freq for freq, token in enumerate(int_to_str)})

		return int_to_str, str_to_int

	def denumericalize(self, indexes):

		if isinstance(indexes, int):

			return self.int_to_str[indexes]

		return [self.int_to_str[index] for index in indexes]

	def numericalize(self, tokens):

		if isinstance(tokens, str):

			return self.str_to_int.get(tokens, self.str_to_int.get(self.unk_token))
		
	 # s = self.str_to_int.get()
			
		return [self.str_to_int.get(token, self.str_to_int.get(self.unk_token)) for token in tokens]

	def __getitem__(self, token):

		return self.str_to_int.get(token, self.str_to_int.get(self.unk_token))

	def __len__(self):
		return len(self.int_to_str)