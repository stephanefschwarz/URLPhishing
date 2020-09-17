import pandas as pd
import numpy as np
import random
import argparse


def command_parser():
	
	parser = argparse.ArgumentParser(description = __doc__)

	parser.add_argument('--input_file', '-i', dest='input_file', 
						required=True, help='Path for the input file.')

	parser.add_argument('--output_file', '-o', dest='output_file', 
						required=True, help='Output file path without name.')

	args = parser.parse_args()

	return args


def main():

	args = command_parser()

	dataset = pd.read_csv(args.input_file)

	unique, counts = np.unique(dataset.domain.astype(str), return_counts=True)

	n_samples = 5000

	samples = random.sample(list(unique), n_samples)

	val_samples = random.sample(samples, n_samples//2)
	val_dataset = dataset[dataset.domain.astype(str).isin(val_samples)]

	test_samples = [samp for samp in samples if not samp in val_samples]
	test_dataset = dataset[dataset.domain.astype(str).isin(test_samples)]

	train_dataset = dataset[~dataset.domain.astype(str).isin(samples)]

	# ================================================================= #
	# 		Saving
	# ================================================================= #

	train_dataset.to_csv(args.output_file + '/train_dataset.csv', index=False)
	test_dataset.to_csv(args.output_file + '/test_dataset.csv', index=False)
	val_dataset.to_csv(args.output_file + '/val_dataset.csv', index=False)

main()
