import pandas as pd
import re
from collections import Counter, defaultdict
from tqdm import tqdm
import pickle
import json
import logging
from logging.config import fileConfig

class Dataset:

	def __init__(self, data_path, tokenizers, numericalizers=dict()):

		raw_data = self._load_data(data_path)

		self.data = self.process_data(raw_data, tokenizers, numericalizers)
 
	def _load_data(self, data_path):

		data = pd.read_csv(data_path)

		data_dict = data.T.to_dict().values()

		return data_dict

	def process_data(self, raw_data, tokenizers, numericalizers=dict()):

		data = []

		for item in tqdm(raw_data):

			_url = dict()
			for key, tokenizer in tokenizers.items():

				numericalizer = numericalizers.get(key)

				if tokenizer is not None:

					_url[key] = tokenizer(item[key])
				
				else:

					_url[key] = item[key]

				if numericalizer is not None:

					_url[key] = numericalizer(_url[key])
			
			data.append(_url)

		return data

	def __getitem__(self, index):

		return self.data[index]

	def __len__(self):

		return len(self.data)

	@staticmethod
	def get_token_frequency(data):

		frequencies = defaultdict(Counter)

		for url in data:

			for item_name, item_value in url.items():

				if(isinstance(item_value, list)):

					frequencies[item_name].update(item_value)

				else:

					frequencies[item_name].update([item_value])

		return frequencies

	@staticmethod
	def to_save(to_save_object, file_path, process_name, type_):

		if (type_ == 'data'):

			with open(file_path, 'wt') as file: # file_path.pkl
				json.dump(to_save_object, file)

		else:

			with open(file_path, 'wb') as file: # file_path.data

				pickle.dump(to_save_object, file, pickle.HIGHEST_PROTOCOL)


	@staticmethod
	def load_file_saved(object_name, process_name, type_):

		if (type_ == 'data'):

			with open(object_name, 'rt') as file:

				obj = json.load(file)

		else:

			with open(object_name, 'rb') as file:

				obj = pickle.load(file)

		return obj

