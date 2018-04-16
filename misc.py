import argparse

import tensorflow as tf
from PIL import Image

def image_subtract_uniform(image, value = (123.68, 116.78, 103.94)):
	channels = tf.split(axis = 2, num_or_size_splits = len(value), value = image)
	for i in range(len(value)):
		channels[i] -= value[i]
	return tf.concat(axis = 2, values = channels)

DEFAULT_LABEL_COLORS = [
	(0, 0, 0),  # 0=background
	# 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
	(128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
	# 6=bus, 7=car, 8=cat, 9=chair, 10=cow
	(0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
	# 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
	(192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
	# 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
	(0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
def decode_labels(mask, label_colors = DEFAULT_LABEL_COLORS):
	img = Image.new('RGB', (len(mask[0]), len(mask)))
	pixels = img.load()
	for j_, j in enumerate(mask[:, :]):
		for k_, k in enumerate(j):
			if k < len(label_colors):
				pixels[k_, j_] = label_colors[k]
	return img

def load_image(filename):
	image_data = tf.read_file(filename)
	image = tf.image.decode_image(image_data)
	image = tf.to_float(tf.image.convert_image_dtype(image, dtype = tf.uint8))
	image.set_shape([None, None, 3])
	return image

def get_common_arguments_parser():
	parser = argparse.ArgumentParser()

	parser.add_argument('--model_dir', type = str, default = './models')

	parser.add_argument('--architecture', type = str, default = 'deeplab_v2')
	parser.add_argument('--base_architecture', type = str, default = 'resnet_v2_101')
	parser.add_argument('--output_stride', type = int, default = 16, choices = [8, 16])

	# params that are used but normally have no custom values
	parser.add_argument('--image_width', type = int, default = 512)
	parser.add_argument('--image_height', type = int, default = 512)
	parser.add_argument('--num_channels', type = int, default = 3)
	parser.add_argument('--num_classes', type = int, default = 21)
	parser.add_argument('--num_training_images', type = int, default = 10582)
	parser.add_argument('--ignore_label', type = int, default = 255)

	parser.add_argument('--batch_norm_decay', type = float, default = 0.9997)

	return parser
