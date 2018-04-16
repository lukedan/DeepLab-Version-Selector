import os
import argparse

import tensorflow as tf
from PIL import Image

import misc
import models

def get_arguments_parser():
	parser = misc.get_common_arguments_parser()

	parser.add_argument('--image_file', type = str)

	return parser

def input_fn(filenames, hparams):
	dataset = tf.data.Dataset.from_tensor_slices(filenames)
	dataset = dataset.map(misc.load_image)
	dataset = dataset.map(misc.image_subtract_uniform)
	dataset = dataset.batch(1)
	return dataset.make_one_shot_iterator().get_next()

def main():
	hparams, unparsed = get_arguments_parser().parse_known_args()
	hparams.batch_size = 1

	model = models.get_estimator(hparams)

	predictions = model.predict(
		input_fn = lambda: input_fn([hparams.image_file], hparams))

	for i, pred in enumerate(predictions):
		misc.decode_labels(pred['classes']).save('test_{}.png'.format(i))

if __name__ == '__main__':
	main()
