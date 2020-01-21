import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
from source.a2c.utils import normalized_columns_initializer

class Network:

	def __init__(self, n_inputs, n_actions, trainer):

		with tf.variable_scope("Network"):

			# Policy network
			self.input_vector = tf.placeholder(shape=[None,n_inputs], dtype=tf.float32)

			self.hidden1 = tf.contrib.layers.fully_connected(self.input_vector, 40, 
																activation_fn=tf.nn.tanh)
			self.hidden2 = tf.contrib.layers.fully_connected(self.hidden1, 32, 
																activation_fn=tf.nn.tanh)
			self.hidden3 = tf.contrib.layers.fully_connected(self.hidden2, 24, 
																activation_fn=tf.nn.tanh)
			self.hidden4 = tf.contrib.layers.fully_connected(self.hidden3, 12, 
																activation_fn=tf.nn.tanh)

			self.policy = tf.contrib.layers.fully_connected(self.hidden4, n_actions,
				activation_fn=tf.nn.softmax,
				weights_initializer=normalized_columns_initializer(0.01),
				biases_initializer=None)

			self.value = tf.contrib.layers.fully_connected(self.hidden4, 1,
				activation_fn=None,
				weights_initializer=normalized_columns_initializer(1.0),
				biases_initializer=None)

			# Training part
			self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
			self.actions_onehot = tf.one_hot(self.actions,n_actions, dtype=tf.float32)
			self.responsible_outputs = tf.reduce_sum(self.policy*self.actions_onehot, [1])

			self.target_value = tf.placeholder(shape=[None], dtype=tf.float32)
			self.advantages = tf.placeholder(shape=[None], dtype=tf.float32)

			# Loss functions
			self.value_loss = 0.5*tf.reduce_sum(tf.square(self.target_value-tf.reshape(self.value,[-1])))
			self.entropy_loss = -tf.reduce_sum(self.policy*tf.log(self.policy))
			self.policy_loss = -tf.reduce_sum(tf.log(self.responsible_outputs)*self.advantages)
			self.total_loss = 0.5*self.value_loss+self.policy_loss-self.entropy_loss*0.01

			# Apply gradients to the network
			self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "Network")
			self.gradients = tf.gradients(self.total_loss, self.variables)
			self.apply_grads = trainer.apply_gradients(zip(self.gradients, self.variables))