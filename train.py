import os
import argparse

import tensorflow as tf

import misc
import models

def get_arguments_parser():
	parser = misc.get_common_arguments_parser()

	parser.add_argument('--data_dir', type = str, default = './dataset')

	parser.add_argument('--batch_size', type = int, default = 10)
	parser.add_argument('--train_epochs', type = int, default = 26)
	parser.add_argument('--epochs_per_eval', type = int, default = 1)
	parser.add_argument('--freeze_batch_norm', action = 'store_true')
	parser.add_argument('--weight_decay', type = float, default = 2e-4)
	parser.add_argument('--learning_rate_policy', type = str, default = 'poly')
	parser.add_argument('--initial_learning_rate', type = float, default = 7e-3)
	parser.add_argument('--end_learning_rate', type = float, default = 1e-6)
	parser.add_argument('--initial_global_step', type = int, default = 0)
	parser.add_argument('--max_iter', type = int, default = 30000)

	parser.add_argument('--power', type = float, default = 0.9)
	parser.add_argument('--momentum', type = float, default = 0.9)

	return parser

def input_fn(is_training, num_epochs, hparams):
	def parse_record(raw_record):
		features = {
			'image/height': tf.FixedLenFeature((), tf.int64),
			'image/width': tf.FixedLenFeature((), tf.int64),
			'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
			'image/format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
			'label/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
			'label/format': tf.FixedLenFeature((), tf.string, default_value='png')
		}
		parsed = tf.parse_single_example(raw_record, features)

		image = tf.image.decode_image(tf.reshape(parsed['image/encoded'], shape = []), hparams.num_channels)
		image = tf.to_float(tf.image.convert_image_dtype(image, dtype = tf.uint8))
		image.set_shape([None, None, hparams.num_channels])

		label = tf.image.decode_image(tf.reshape(parsed['label/encoded'], shape = []), 1)
		label = tf.to_int32(tf.image.convert_image_dtype(label, dtype = tf.uint8))
		label.set_shape([None, None, 1])

		return image, label

	def preprocess_image(image, label):
		def random_crop_or_pad_image_and_label(image, label):
			label = label - hparams.ignore_label
			label = tf.to_float(label)
			image_height = tf.shape(image)[0]
			image_width = tf.shape(image)[1]
			image_and_label = tf.concat([image, label], axis = 2)
			image_and_label_pad = tf.image.pad_to_bounding_box(
				image_and_label, 0, 0,
				tf.maximum(hparams.image_height, image_height),
				tf.maximum(hparams.image_width, image_width))
			image_and_label_crop = tf.random_crop(
				image_and_label_pad, [hparams.image_height, hparams.image_width, hparams.num_channels + 1])

			image_crop = image_and_label_crop[:, :, :hparams.num_channels]
			label_crop = image_and_label_crop[:, :, hparams.num_channels:]
			label_crop += hparams.ignore_label
			label_crop = tf.to_int32(label_crop)

			return image_crop, label_crop

		if is_training:
			# TODO rescale & flip
			image, label = random_crop_or_pad_image_and_label(image, label)
			image.set_shape([hparams.image_height, hparams.image_width, hparams.num_channels])
			label.set_shape([hparams.image_height, hparams.image_width, 1])

		image = misc.image_subtract_uniform(image)

		return image, label

	filenames = [os.path.join(hparams.data_dir, 'voc_train.record' if is_training else 'voc_val.record')]

	dataset = tf.data.Dataset.from_tensor_slices(filenames).flat_map(tf.data.TFRecordDataset)
	if is_training:
		dataset = dataset.shuffle(buffer_size = hparams.num_training_images)
	dataset = dataset.map(parse_record)
	dataset = dataset.map(preprocess_image)
	dataset = dataset.prefetch(hparams.batch_size)

	dataset = dataset.repeat(num_epochs).batch(hparams.batch_size)
	return dataset.make_one_shot_iterator().get_next()

def main():
	hparams, unparsed = get_arguments_parser().parse_known_args()

	model = models.get_estimator(
		hparams, config = tf.estimator.RunConfig().replace(save_checkpoints_secs = 1e9))

	tensors_to_log = {
		'learning_rate': 'learning_rate',
		'cross_entropy': 'cross_entropy'
	}
	logging_hook = tf.train.LoggingTensorHook(tensors = tensors_to_log, every_n_iter = 1)

	for i in range(hparams.train_epochs // hparams.epochs_per_eval):
		print 'train step'
		model.train(
			input_fn = lambda: input_fn(True, hparams.epochs_per_eval, hparams),
			hooks = [logging_hook])

		print 'eval step'
		eval_resutls = model.evaluate(input_fn = lambda: input_fn(False, 1, hparams))
		print eval_resutls

if __name__ == '__main__':
	main()
