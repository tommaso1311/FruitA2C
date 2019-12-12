import tensorflow as tf
import numpy as np
from scipy.signal import lfilter

def normalized_columns_initializer(std=1.0):
	"""
	Creates an initializer using a normal distribution

	Parameters
	----------
	std : float
		standard deviaton of the distribution

	Returns
	-------
	_initializer : function
		initializer function
	"""
	def _initializer(shape, dtype=None, partition_info=None):
		out = np.random.randn(*shape).astype(np.float32)
		out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
		return tf.constant(out)
	return _initializer

def discount(x, gamma):
	"""
	Discounts an array of values

	Parameters
	----------
	x : array
		array of values
	gamma : float
		discount parameter
	"""
	return lfilter([1], [1, -gamma], x[::-1], axis=0)[::-1]

class consts:

	SAME = "same"
	# WAIT = "wait"
	DIFFERENT = "different"

	ACTIONS = [SAME, DIFFERENT]

	CORRECT_GUESS_REWARD = +1
	WRONG_GUESS_REWARD = -1