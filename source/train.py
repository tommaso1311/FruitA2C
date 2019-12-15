import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
import tensorflow as tf

from source.a2c.network import Network
from source.a2c.agent import Agent

def run(n_inputs, n_actions, model_path, load_model, gamma, epsilon,
		starting_index, final_index, buffer_length, graphs_step):

	tf.reset_default_graph()

	if not os.path.exists(model_path):
		os.makedirs(model_path)

	with tf.device("/gpu:0"):

		trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
		agent = Agent(n_inputs, n_actions, trainer, model_path)
		saver = tf.train.Saver(max_to_keep=5)

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
			log_device_placement=True)) as sess:

		if load_model == True:
			ckpt = tf.train.get_checkpoint_state(model_path)
			saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			sess.run(tf.global_variables_initializer())

		agent.train(sess, gamma, epsilon, saver,
					starting_index, final_index, buffer_length, graphs_step)