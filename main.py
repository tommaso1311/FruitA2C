import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import source.train as train
from source.a2c.network import Network
from source.a2c.agent import Agent
# import source.test as test
from os import path
import pandas as pd

model_path = "./model"

n_inputs = 7
n_actions = 2

starting_index = 50000
final_index = 250000
step = 1000

gamma = 0.1
epsilon = 0.0

buffer_length = False
graphs_step = 5

# test_games = 10
# save_games = True

abs_path = os.getcwd()
data_path = path.join(abs_path, "dataset")
ready_path = path.join(data_path, "ready")

def main():
	load_model = bool(starting_index)

	tf.reset_default_graph()

	if not os.path.exists(model_path):
		os.makedirs(model_path)

	with tf.device("/gpu:0"):

		trainer = tf.train.AdamOptimizer(learning_rate=1e-4)
		agent = Agent(n_inputs, n_actions, trainer, model_path)
		saver = tf.train.Saver(max_to_keep=5)

	with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
		for i in range(starting_index, final_index, step):

			dataset = path.join(ready_path, f"defects_generated_{i}_{i+step-1}.csv")
			df = pd.read_csv(dataset)
		
			if load_model == True:
				ckpt = tf.train.get_checkpoint_state(model_path)
				saver.restore(sess, ckpt.model_checkpoint_path)
			else:
				sess.run(tf.global_variables_initializer())

			agent.train(sess, gamma, epsilon, saver,
						i, i+step, buffer_length, graphs_step, df)

			load_model = True

if __name__ == "__main__":
	main()