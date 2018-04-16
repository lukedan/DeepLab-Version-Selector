import os

import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v2
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers

def deeplab_v2(inputs, labels, mode, hparams):
	def layer(op):
		'''Decorator for composable network layers.'''

		def layer_decorated(self, *args, **kwargs):
			# Automatically set a name if not provided.
			name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
			# Figure out the layer inputs.
			if len(self.terminals) == 0:
				raise RuntimeError('No input variables found for layer %s.' % name)
			elif len(self.terminals) == 1:
				layer_input = self.terminals[0]
			else:
				layer_input = list(self.terminals)
			# Perform the operation and get the output.
			layer_output = op(self, layer_input, *args, **kwargs)
			# Add to layer LUT.
			self.layers[name] = layer_output
			# This output is now the input for the next layer.
			self.feed(layer_output)
			# Return self for chained calls.
			return self

		return layer_decorated

	class Network:
		def __init__(self, inputs, trainable=True, is_training=False, num_classes=21):
			# The input nodes for this network
			self.inputs = inputs
			# The current list of terminal nodes
			self.terminals = []
			# Mapping from layer names to layers
			self.layers = dict(inputs)
			# If true, the resulting variables are set as trainable
			self.trainable = trainable
			# Switch variable for dropout
			self.use_dropout = tf.placeholder_with_default(tf.constant(1.0),
														shape=[],
														name='use_dropout')
			self.setup(is_training, num_classes)

		def setup(self, is_training, num_classes):
			(self.feed('data')
				.conv(7, 7, 64, 2, 2, biased=False, relu=False, name='conv1')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn_conv1')
				.max_pool(3, 3, 2, 2, name='pool1')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch1')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch1'))

			(self.feed('pool1')
				.conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2a_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2a')
				.conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2a_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2a_branch2b')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2a_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn2a_branch2c'))

			(self.feed('bn2a_branch1',
					'bn2a_branch2c')
				.add(name='res2a')
				.relu(name='res2a_relu')
				.conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2b_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2a')
				.conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2b_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2b_branch2b')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2b_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn2b_branch2c'))

			(self.feed('res2a_relu',
					'bn2b_branch2c')
				.add(name='res2b')
				.relu(name='res2b_relu')
				.conv(1, 1, 64, 1, 1, biased=False, relu=False, name='res2c_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2a')
				.conv(3, 3, 64, 1, 1, biased=False, relu=False, name='res2c_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn2c_branch2b')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res2c_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn2c_branch2c'))

			(self.feed('res2b_relu',
					'bn2c_branch2c')
				.add(name='res2c')
				.relu(name='res2c_relu')
				.conv(1, 1, 512, 2, 2, biased=False, relu=False, name='res3a_branch1')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch1'))

			(self.feed('res2c_relu')
				.conv(1, 1, 128, 2, 2, biased=False, relu=False, name='res3a_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2a')
				.conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3a_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3a_branch2b')
				.conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3a_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn3a_branch2c'))

			(self.feed('bn3a_branch1',
					'bn3a_branch2c')
				.add(name='res3a')
				.relu(name='res3a_relu')
				.conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2a')
				.conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b1_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b1_branch2b')
				.conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b1_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn3b1_branch2c'))

			(self.feed('res3a_relu',
					'bn3b1_branch2c')
				.add(name='res3b1')
				.relu(name='res3b1_relu')
				.conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2a')
				.conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b2_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b2_branch2b')
				.conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b2_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn3b2_branch2c'))

			(self.feed('res3b1_relu',
					'bn3b2_branch2c')
				.add(name='res3b2')
				.relu(name='res3b2_relu')
				.conv(1, 1, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2a')
				.conv(3, 3, 128, 1, 1, biased=False, relu=False, name='res3b3_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn3b3_branch2b')
				.conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res3b3_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn3b3_branch2c'))

			(self.feed('res3b2_relu',
					'bn3b3_branch2c')
				.add(name='res3b3')
				.relu(name='res3b3_relu')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch1')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch1'))

			(self.feed('res3b3_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4a_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4a_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4a_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4a_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4a_branch2c'))

			(self.feed('bn4a_branch1',
					'bn4a_branch2c')
				.add(name='res4a')
				.relu(name='res4a_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b1_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b1_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b1_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b1_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b1_branch2c'))

			(self.feed('res4a_relu',
					'bn4b1_branch2c')
				.add(name='res4b1')
				.relu(name='res4b1_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b2_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b2_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b2_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b2_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b2_branch2c'))

			(self.feed('res4b1_relu',
					'bn4b2_branch2c')
				.add(name='res4b2')
				.relu(name='res4b2_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b3_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b3_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b3_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b3_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b3_branch2c'))

			(self.feed('res4b2_relu',
					'bn4b3_branch2c')
				.add(name='res4b3')
				.relu(name='res4b3_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b4_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b4_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b4_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b4_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b4_branch2c'))

			(self.feed('res4b3_relu',
					'bn4b4_branch2c')
				.add(name='res4b4')
				.relu(name='res4b4_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b5_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b5_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b5_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b5_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b5_branch2c'))

			(self.feed('res4b4_relu',
					'bn4b5_branch2c')
				.add(name='res4b5')
				.relu(name='res4b5_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b6_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b6_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b6_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b6_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b6_branch2c'))

			(self.feed('res4b5_relu',
					'bn4b6_branch2c')
				.add(name='res4b6')
				.relu(name='res4b6_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b7_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b7_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b7_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b7_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b7_branch2c'))

			(self.feed('res4b6_relu',
					'bn4b7_branch2c')
				.add(name='res4b7')
				.relu(name='res4b7_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b8_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b8_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b8_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b8_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b8_branch2c'))

			(self.feed('res4b7_relu',
					'bn4b8_branch2c')
				.add(name='res4b8')
				.relu(name='res4b8_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b9_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b9_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b9_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b9_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b9_branch2c'))

			(self.feed('res4b8_relu',
					'bn4b9_branch2c')
				.add(name='res4b9')
				.relu(name='res4b9_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b10_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b10_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b10_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b10_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b10_branch2c'))

			(self.feed('res4b9_relu',
					'bn4b10_branch2c')
				.add(name='res4b10')
				.relu(name='res4b10_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b11_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b11_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b11_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b11_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b11_branch2c'))

			(self.feed('res4b10_relu',
					'bn4b11_branch2c')
				.add(name='res4b11')
				.relu(name='res4b11_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b12_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b12_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b12_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b12_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b12_branch2c'))

			(self.feed('res4b11_relu',
					'bn4b12_branch2c')
				.add(name='res4b12')
				.relu(name='res4b12_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b13_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b13_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b13_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b13_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b13_branch2c'))

			(self.feed('res4b12_relu',
					'bn4b13_branch2c')
				.add(name='res4b13')
				.relu(name='res4b13_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b14_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b14_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b14_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b14_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b14_branch2c'))

			(self.feed('res4b13_relu',
					'bn4b14_branch2c')
				.add(name='res4b14')
				.relu(name='res4b14_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b15_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b15_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b15_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b15_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b15_branch2c'))

			(self.feed('res4b14_relu',
					'bn4b15_branch2c')
				.add(name='res4b15')
				.relu(name='res4b15_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b16_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b16_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b16_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b16_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b16_branch2c'))

			(self.feed('res4b15_relu',
					'bn4b16_branch2c')
				.add(name='res4b16')
				.relu(name='res4b16_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b17_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b17_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b17_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b17_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b17_branch2c'))

			(self.feed('res4b16_relu',
					'bn4b17_branch2c')
				.add(name='res4b17')
				.relu(name='res4b17_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b18_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b18_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b18_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b18_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b18_branch2c'))

			(self.feed('res4b17_relu',
					'bn4b18_branch2c')
				.add(name='res4b18')
				.relu(name='res4b18_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b19_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b19_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b19_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b19_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b19_branch2c'))

			(self.feed('res4b18_relu',
					'bn4b19_branch2c')
				.add(name='res4b19')
				.relu(name='res4b19_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b20_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b20_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b20_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b20_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b20_branch2c'))

			(self.feed('res4b19_relu',
					'bn4b20_branch2c')
				.add(name='res4b20')
				.relu(name='res4b20_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b21_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b21_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b21_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b21_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b21_branch2c'))

			(self.feed('res4b20_relu',
					'bn4b21_branch2c')
				.add(name='res4b21')
				.relu(name='res4b21_relu')
				.conv(1, 1, 256, 1, 1, biased=False, relu=False, name='res4b22_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2a')
				.atrous_conv(3, 3, 256, 2, padding='SAME', biased=False, relu=False, name='res4b22_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn4b22_branch2b')
				.conv(1, 1, 1024, 1, 1, biased=False, relu=False, name='res4b22_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn4b22_branch2c'))

			(self.feed('res4b21_relu',
					'bn4b22_branch2c')
				.add(name='res4b22')
				.relu(name='res4b22_relu')
				.conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch1')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch1'))

			(self.feed('res4b22_relu')
				.conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5a_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2a')
				.atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5a_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5a_branch2b')
				.conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5a_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn5a_branch2c'))

			(self.feed('bn5a_branch1',
					'bn5a_branch2c')
				.add(name='res5a')
				.relu(name='res5a_relu')
				.conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5b_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2a')
				.atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5b_branch2b')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5b_branch2b')
				.conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5b_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn5b_branch2c'))

			(self.feed('res5a_relu',
					'bn5b_branch2c')
				.add(name='res5b')
				.relu(name='res5b_relu')
				.conv(1, 1, 512, 1, 1, biased=False, relu=False, name='res5c_branch2a')
				.batch_normalization(is_training=is_training, activation_fn=tf.nn.relu, name='bn5c_branch2a')
				.atrous_conv(3, 3, 512, 4, padding='SAME', biased=False, relu=False, name='res5c_branch2b')
				.batch_normalization(activation_fn=tf.nn.relu, name='bn5c_branch2b', is_training=is_training)
				.conv(1, 1, 2048, 1, 1, biased=False, relu=False, name='res5c_branch2c')
				.batch_normalization(is_training=is_training, activation_fn=None, name='bn5c_branch2c'))

			(self.feed('res5b_relu',
					'bn5c_branch2c')
				.add(name='res5c')
				.relu(name='res5c_relu')
				.atrous_conv(3, 3, num_classes, 6, padding='SAME', relu=False, name='fc1_voc12_c0'))

			(self.feed('res5c_relu')
				.atrous_conv(3, 3, num_classes, 12, padding='SAME', relu=False, name='fc1_voc12_c1'))

			(self.feed('res5c_relu')
				.atrous_conv(3, 3, num_classes, 18, padding='SAME', relu=False, name='fc1_voc12_c2'))

			(self.feed('res5c_relu')
				.atrous_conv(3, 3, num_classes, 24, padding='SAME', relu=False, name='fc1_voc12_c3'))

			(self.feed('fc1_voc12_c0',
					'fc1_voc12_c1',
					'fc1_voc12_c2',
					'fc1_voc12_c3')
				.add(name='fc1_voc12'))

		def feed(self, *args):
			'''Set the input(s) for the next operation by replacing the terminal nodes.
			The arguments can be either layer names or the actual layers.
			'''
			assert len(args) != 0
			self.terminals = []
			for fed_layer in args:
				if isinstance(fed_layer, basestring):
					try:
						fed_layer = self.layers[fed_layer]
					except KeyError:
						raise KeyError('Unknown layer name fed: %s' % fed_layer)
				self.terminals.append(fed_layer)
			return self

		def get_output(self):
			'''Returns the current network output.'''
			return self.terminals[-1]

		def get_unique_name(self, prefix):
			'''Returns an index-suffixed unique name for the given prefix.
			This is used for auto-generating layer names based on the type-prefix.
			'''
			ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
			return '%s_%d' % (prefix, ident)

		def make_var(self, name, shape):
			'''Creates a new TensorFlow variable.'''
			return tf.get_variable(name, shape, trainable=self.trainable)

		def validate_padding(self, padding):
			'''Verifies that the padding is one of the supported ones.'''
			assert padding in ('SAME', 'VALID')

		@layer
		def conv(self,
				input,
				k_h,
				k_w,
				c_o,
				s_h,
				s_w,
				name,
				relu=True,
				padding='SAME',
				group=1,
				biased=True):
			# Verify that the padding is acceptable
			self.validate_padding(padding)
			# Get the number of channels in the input
			c_i = input.get_shape()[-1]
			# Verify that the grouping parameter is valid
			assert c_i % group == 0
			assert c_o % group == 0
			# Convolution for a given input and kernel
			convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
			with tf.variable_scope(name) as scope:
				kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
				if group == 1:
					# This is the common-case. Convolve the input without any further complications.
					output = convolve(input, kernel)
				else:
					# Split the input into groups and then convolve each of them independently
					input_groups = tf.split(3, group, input)
					kernel_groups = tf.split(3, group, kernel)
					output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
					# Concatenate the groups
					output = tf.concat(3, output_groups)
				# Add the biases
				if biased:
					biases = self.make_var('biases', [c_o])
					output = tf.nn.bias_add(output, biases)
				if relu:
					# ReLU non-linearity
					output = tf.nn.relu(output, name=scope.name)
				return output

		@layer
		def atrous_conv(self,
						input,
						k_h,
						k_w,
						c_o,
						dilation,
						name,
						relu=True,
						padding='SAME',
						group=1,
						biased=True):
			# Verify that the padding is acceptable
			self.validate_padding(padding)
			# Get the number of channels in the input
			c_i = input.get_shape()[-1]
			# Verify that the grouping parameter is valid
			assert c_i % group == 0
			assert c_o % group == 0
			# Convolution for a given input and kernel
			convolve = lambda i, k: tf.nn.atrous_conv2d(i, k, dilation, padding=padding)
			with tf.variable_scope(name) as scope:
				kernel = self.make_var('weights', shape=[k_h, k_w, c_i / group, c_o])
				if group == 1:
					# This is the common-case. Convolve the input without any further complications.
					output = convolve(input, kernel)
				else:
					# Split the input into groups and then convolve each of them independently
					input_groups = tf.split(3, group, input)
					kernel_groups = tf.split(3, group, kernel)
					output_groups = [convolve(i, k) for i, k in zip(input_groups, kernel_groups)]
					# Concatenate the groups
					output = tf.concat(3, output_groups)
				# Add the biases
				if biased:
					biases = self.make_var('biases', [c_o])
					output = tf.nn.bias_add(output, biases)
				if relu:
					# ReLU non-linearity
					output = tf.nn.relu(output, name=scope.name)
				return output

		@layer
		def relu(self, input, name):
			return tf.nn.relu(input, name=name)

		@layer
		def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding='SAME'):
			self.validate_padding(padding)
			return tf.nn.max_pool(input,
								ksize=[1, k_h, k_w, 1],
								strides=[1, s_h, s_w, 1],
								padding=padding,
								name=name)

		@layer
		def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding='SAME'):
			self.validate_padding(padding)
			return tf.nn.avg_pool(input,
								ksize=[1, k_h, k_w, 1],
								strides=[1, s_h, s_w, 1],
								padding=padding,
								name=name)

		@layer
		def lrn(self, input, radius, alpha, beta, name, bias=1.0):
			return tf.nn.local_response_normalization(input,
													depth_radius=radius,
													alpha=alpha,
													beta=beta,
													bias=bias,
													name=name)

		@layer
		def concat(self, inputs, axis, name):
			return tf.concat(concat_dim=axis, values=inputs, name=name)

		@layer
		def add(self, inputs, name):
			return tf.add_n(inputs, name=name)

		@layer
		def fc(self, input, num_out, name, relu=True):
			with tf.variable_scope(name) as scope:
				input_shape = input.get_shape()
				if input_shape.ndims == 4:
					# The input is spatial. Vectorize it first.
					dim = 1
					for d in input_shape[1:].as_list():
						dim *= d
					feed_in = tf.reshape(input, [-1, dim])
				else:
					feed_in, dim = (input, input_shape[-1].value)
				weights = self.make_var('weights', shape=[dim, num_out])
				biases = self.make_var('biases', [num_out])
				op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
				fc = op(feed_in, weights, biases, name=scope.name)
				return fc

		@layer
		def softmax(self, input, name):
			input_shape = map(lambda v: v.value, input.get_shape())
			if len(input_shape) > 2:
				# For certain models (like NiN), the singleton spatial dimensions
				# need to be explicitly squeezed, since they're not broadcast-able
				# in TensorFlow's NHWC ordering (unlike Caffe's NCHW).
				if input_shape[1] == 1 and input_shape[2] == 1:
					input = tf.squeeze(input, squeeze_dims=[1, 2])
				else:
					raise ValueError('Rank 2 tensor input expected for softmax!')
			return tf.nn.softmax(input, name)

		@layer
		def batch_normalization(self, input, name, is_training, activation_fn=None, scale=True):
			with tf.variable_scope(name) as scope:
				output = tf.contrib.slim.batch_norm(
					input,
					activation_fn=activation_fn,
					is_training=is_training,
					updates_collections=None,
					scale=scale,
					scope=scope)
				return output

		@layer
		def dropout(self, input, keep_prob, name):
			keep = 1 - self.use_dropout + (self.use_dropout * keep_prob)
			return tf.nn.dropout(input, keep, name=name)

	net = Network(
		{'data': inputs},
		is_training = mode == tf.estimator.ModeKeys.TRAIN,
		num_classes = hparams.num_classes)
	raw_output = net.layers['fc1_voc12']
	return raw_output, tf.image.resize_nearest_neighbor(labels, tf.shape(raw_output)[1:3])


def deeplab_v3plus(inputs, labels, mode, hparams):
	is_training = mode == tf.estimator.ModeKeys.TRAIN

	def atrous_spatial_pyramid_pooling(aspp_input, depth = 256):
		with tf.variable_scope('aspp'):
			atrous_rates = [6, 12, 18]
			if hparams.output_stride == 8:
				atrous_rates = [2 * rate for rate in atrous_rates]
			with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay = hparams.batch_norm_decay)):
				with arg_scope([layers.batch_norm], is_training = is_training):
					inputs_size = tf.shape(aspp_input)[1:3]
					conv_1x1 = tf.contrib.layers.conv2d(
						aspp_input, depth, [1, 1], stride = 1, scope = "conv_1x1")
					conv_3x3_1 = tf.contrib.layers.conv2d(
						aspp_input, depth, [3, 3], stride = 1, rate = atrous_rates[0], scope = 'conv_3x3_1')
					conv_3x3_2 = tf.contrib.layers.conv2d(
						aspp_input, depth, [3, 3], stride = 1, rate = atrous_rates[1], scope = 'conv_3x3_2')
					conv_3x3_3 = tf.contrib.layers.conv2d(
						aspp_input, depth, [3, 3], stride = 1, rate = atrous_rates[2], scope = 'conv_3x3_3')
					with tf.variable_scope('image_level_features'):
						image_level_features = tf.reduce_mean(
							aspp_input, [1, 2], name = 'global_average_pooling', keepdims = True)
						image_level_features = tf.contrib.layers.conv2d(
							image_level_features, depth, [1, 1], stride = 1, scope = 'conv_1x1')
						image_level_features = tf.image.resize_bilinear(
							image_level_features, inputs_size, name = 'upsample')
					net = tf.concat(
						[conv_1x1, conv_3x3_1, conv_3x3_2, conv_3x3_3, image_level_features],
						axis = 3, name = 'concat')
					return tf.contrib.layers.conv2d(net, depth, [1, 1], stride = 1, scope = 'conv_1x1_concat')

	base_model = getattr(resnet_v2, hparams.base_architecture)
	# inputs = tf.transpose(inputs, [0, 3, 1, 2])

	# encoder
	with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay = hparams.batch_norm_decay)):
		logits, end_points = base_model(
			inputs,
			num_classes = None,
			is_training = is_training,
			global_pool = False,
			output_stride = hparams.output_stride)
	if is_training: # load model
		variables_to_restore = tf.contrib.slim.get_variables_to_restore(
			exclude = [hparams.base_architecture + '/logits', 'global_step'])
		tf.train.init_from_checkpoint(
			hparams.pre_trained_model, {v.name.split(':')[0]: v for v in variables_to_restore})
	inputs_size = tf.shape(inputs)[1:3]
	net = end_points[hparams.base_architecture + '/block4']
	encoder_output = atrous_spatial_pyramid_pooling(net)

	# decoder
	with tf.variable_scope('decoder'):
		with tf.contrib.slim.arg_scope(resnet_v2.resnet_arg_scope(batch_norm_decay = hparams.batch_norm_decay)):
			with arg_scope([layers.batch_norm], is_training = is_training):
				with tf.variable_scope("low_level_features"):
					low_level_features = end_points[hparams.base_architecture + '/block1/unit_3/bottleneck_v2/conv1']
					low_level_features = tf.contrib.layers.conv2d(
						low_level_features,
						num_outputs = 48,
						kernel_size = [1, 1],
						stride = 1,
						scope = 'conv_1x1')
					low_level_features_size = tf.shape(low_level_features)[1:3]
				with tf.variable_scope('upsampling_logits'):
					net = tf.image.resize_bilinear(encoder_output, low_level_features_size, name = 'upsample_1')
					net = tf.concat([net, low_level_features], axis = 3, name = 'concat')
					net = tf.contrib.layers.conv2d(net, 256, [3, 3], stride = 1, scope = 'conv_3x3_1')
					net = tf.contrib.layers.conv2d(net, 256, [3, 3], stride = 1, scope = 'conv_3x3_2')
					net = tf.contrib.layers.conv2d(
						net, hparams.num_classes, [1, 1],
						activation_fn = None, normalizer_fn = None, scope = 'conv_1x1')
					return tf.image.resize_bilinear(net, inputs_size, name = 'upsample_2'), labels


def get_estimator(hparams, **kwargs):
	model_fn = globals()[hparams.architecture]

	def get_estimator_spec(inputs, labels, mode):
		outputs, labels = model_fn(inputs, labels, mode, hparams)

		pred_classes = tf.argmax(outputs, axis = 3, output_type = tf.int32)
		predictions = {
			'classes': pred_classes,
			'probabilities': tf.nn.softmax(outputs, name = 'softmax_tensor')
		}

		loss = None
		train_op = None

		if mode != tf.estimator.ModeKeys.PREDICT:
			logits_flat = tf.reshape(outputs, [-1, hparams.num_classes])
			labels_flat = tf.reshape(labels, [-1, ])

			valid_indices = tf.to_int32(labels_flat < hparams.num_classes)
			valid_logits = tf.dynamic_partition(logits_flat, valid_indices, num_partitions = 2)[1]
			valid_labels = tf.dynamic_partition(labels_flat, valid_indices, num_partitions = 2)[1]

			cross_entropy = tf.losses.sparse_softmax_cross_entropy(logits = valid_logits, labels = valid_labels)
			tf.identity(cross_entropy, name = 'cross_entropy')
			tf.summary.scalar('cross_entropy', cross_entropy)

			train_var_list = tf.trainable_variables()
			if hparams.freeze_batch_norm:
				train_var_list = [
					v for v in train_var_list if 'beta' not in v.name and 'gamma' not in v.name]
			with tf.variable_scope('total_loss'):
				loss = cross_entropy + hparams.weight_decay * tf.add_n([tf.nn.l2_loss(v) for v in train_var_list])

			if mode == tf.estimator.ModeKeys.TRAIN:
				global_step = tf.train.get_or_create_global_step()
				if hparams.learning_rate_policy == 'piecewise':
					learning_rate = tf.train.piecewise_constant(
						tf.cast(global_step, tf.int32),
						[int(epoch * hparams.num_train / hparams.batch_size) for epoch in (100, 150, 200)],
						[decay * 0.1 * hparams.batch_size / 128 for decay in (1, 0.1, 0.01, 0.001)])
				elif hparams.learning_rate_policy == 'poly':
					learning_rate = tf.train.polynomial_decay(
						hparams.initial_learning_rate,
						tf.cast(global_step, tf.int32) - hparams.initial_global_step,
						hparams.max_iter, hparams.end_learning_rate, hparams.power)
				tf.identity(learning_rate, name = 'learning_rate')
				tf.summary.scalar('learning_rate', learning_rate)
				optimizer = tf.train.MomentumOptimizer(learning_rate = learning_rate, momentum = hparams.momentum)

				with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
					train_op = optimizer.minimize(loss, global_step, var_list = train_var_list)

		return tf.estimator.EstimatorSpec(
			mode = mode,
			predictions = predictions,
			loss = loss,
			train_op = train_op)

	return tf.estimator.Estimator(
		model_fn = lambda features, labels, mode, params: get_estimator_spec(features, labels, mode),
		model_dir = os.path.join(hparams.model_dir, hparams.architecture),
		params = hparams,
		**kwargs)
