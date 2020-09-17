import pandas as pd
import numpy as np
import tldextract
import argparse
import whois
import tldextract
import math
import re
from datetime import date

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

def get_url_data_from_whois(url):

	# days | http | https | www
	url_structure = tldextract.extract(url)
	print(url_structure)
	url_domain = url_structure.domain + '.' + url_structure.suffix
	print(url_domain)
	try:

		whois_response = whois.query(url_domain).__dict__
		domain =  whois_response['name']
		domain_creation = whois_response['creation_date']
		data_in_days = date.today() - whois_response['creation_date'].date()
		data_in_days = data_in_days.days

		print('Try OK, days: ', data_in_days)
	except Exception as e:

		print('Could not get url whois: ', e)
		data_in_days = 0

	http = 1 if re.search('^http:', url) else 0
	https = 1 if re.search('^https:', url) else 0
	www = 1 if re.search('^www:', url) else 0

	return [data_in_days, http, https, www]

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



