import pandas as pd
import numpy as np
import tldextract
import argparse

def command_parser():
	
	parser = argparse.ArgumentParser(description = __doc__)

	parser.add_argument('--input_file', '-i', dest='input_file', 
						required=True, help='Path for the input file.')

	parser.add_argument('--column_name', '-c', dest='column_name', 
						required=False, default=None, help='URL column name.')

	parser.add_argument('--output_file', '-o', dest='output_file', 
						required=True, help='Output file path.')

	args = parser.parse_args()

	return args

def remove_unicode(url):

	#dataset['new'] = dataset.url.apply(lambda url: url.encode('ascii', 'ignore').decode('ascii'))


	return url.encode('ascii', 'ignore').decode('ascii')


def get_url_domain(url):
	
	request = tldextract.extract(url)

	return request.domain


def main():
	
	args = command_parser()

	dataset = pd.read_csv(args.input_file)


	if (args.column_name == None):

		dataset['url'] = dataset.url.apply(lambda url: remove_unicode(url))

		dataset['domain'] = dataset.url.apply(lambda url: get_url_domain(url))

	else:

		dataset[eval(args.column_name)] = dataset.eval(args.column_name).apply(lambda url: remove_unicode(url))

		dataset['domain'] = dataset.eval(args.column_name).apply(lambda url: get_url_domain(url))


	dataset.to_csv(args.output_file, index=False)


main()
