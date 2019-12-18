import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import source.train as train
# import source.test as test

model_path = "./model"

n_inputs = 7
n_actions = 2

starting_index = 0
final_index = 100
step = 10

gamma = 0.99
epsilon = 0.0

buffer_length = False
graphs_step = 5

# test_games = 10
# save_games = True

def main():
	for i in range(starting_index, final_index, step):
		train.run(n_inputs, n_actions, model_path, bool(i), gamma, epsilon,
				i, i+step, buffer_length, graphs_step)
	# test.run(n_inputs, n_actions, model_path, 0.0, test_games, save_games)

if __name__ == "__main__":
	main()