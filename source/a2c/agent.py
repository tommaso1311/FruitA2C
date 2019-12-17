from source.fruit.fruit import Fruit
from source.a2c import utils
from source.a2c.network import Network
import tensorflow as tf
import numpy as np
import random
import os

class Agent:

	def __init__(self, n_inputs, n_actions, trainer, model_path):

		self.model_path = model_path
		self.actions_available = utils.consts.ACTIONS
		self.network = Network(n_inputs, n_actions, trainer)
		self.summary_writer = tf.summary.FileWriter("./graphs")

	def policy(self, sess, input_vector, epsilon):

		np.random.seed()
		c = np.random.rand()
		
		actions_distribution = sess.run(self.network.policy,
										feed_dict={self.network.input_vector:input_vector})
		if c < 1-epsilon:
			action_idx = np.random.choice(len(actions_distribution[0]), p=actions_distribution[0])
		else:
			action_idx = np.random.choice(len(actions_distribution[0]))

		return self.actions_available[action_idx], action_idx

	def value(self, sess, input_vector):

		value = sess.run(self.network.value,
						feed_dict={self.network.input_vector:input_vector})

		return value[0, 0]

	def update(self, sess, fruit_buffer, gamma, bootstrap_value):

		fruit_buffer = np.array(fruit_buffer)

		states = fruit_buffer[:, 0]
		actions = fruit_buffer[:, 1]
		rewards = fruit_buffer[:, 2]
		values = fruit_buffer[:, 3]

		rewards_plus = np.asarray(rewards.tolist() + [bootstrap_value])
		discounted_rewards = utils.discount(rewards_plus, gamma)[:-1]

		values_plus = np.asarray(values.tolist() + [bootstrap_value])
		advantages = rewards + gamma * values_plus[1:] - values_plus[:-1]
		advantages = utils.discount(advantages, gamma)

		feed_dict = {self.network.target_value:discounted_rewards,
					self.network.input_vector:np.vstack(states),
					self.network.actions:actions,
					self.network.advantages:advantages}
		value_loss, policy_loss, entropy_loss, total_loss, _ = sess.run([self.network.value_loss,
															self.network.policy_loss,
															self.network.entropy_loss,
															self.network.total_loss,
															self.network.apply_grads],
															feed_dict=feed_dict)

		length = len(fruit_buffer)
		return value_loss/length, policy_loss/length, entropy_loss/length, total_loss/length

	def train(self, sess, gamma, epsilon, saver,
				fruits_analyzed, max_fruits_analyzed,
				buffer_length=5, graphs_step=5):

		with sess.as_default(), sess.graph.as_default():
			while fruits_analyzed < max_fruits_analyzed:

				# fruit = Fruit.from_file(fruits_analyzed)
				fruit = Fruit.online(fruits_analyzed)

				print(f"Analyzing {fruits_analyzed} over {max_fruits_analyzed}", end="\r", flush=True)

				fruit_buffer = []
				fruit_values = []
				fruit_rewards = []

				while fruit.is_analyzable():

					fruit.update_current_defect()

					for defect in fruit.get_defects_analyzed():

						state = fruit.get_state(defect)
						action, action_idx = self.policy(sess, state, epsilon)
						value = self.value(sess, state)
						reward = fruit.add_guess(defect, action)

						fruit_buffer.append([state, action_idx, reward, value])
						fruit_values.append(value)
						fruit_rewards.append(reward)

					fruit.apply_UUID()

				if len(fruit_buffer) != 0:
					fruit_avg_reward = np.mean(fruit_rewards)
					fruit_avg_value = np.mean(fruit_values)

				if len(fruit_buffer) != 0:
					v_l, p_l, e_l, t_l = self.update(sess, fruit_buffer, gamma, 0.0)

				if fruits_analyzed % 5 == 0 and fruits_analyzed != 0 and len(fruit_buffer) != 0:

					summary = tf.Summary()
					summary.value.add(tag='Losses/Value Loss', simple_value=float(v_l))
					summary.value.add(tag='Losses/Policy Loss', simple_value=float(p_l))
					summary.value.add(tag='Losses/Entropy', simple_value=float(e_l))
					summary.value.add(tag='Losses/Total Loss', simple_value=float(t_l))
					summary.value.add(tag='Performances/Fruit Average Reward',
										simple_value=float(fruit_avg_reward))
					summary.value.add(tag='Performances/Fruit Average State Value',
										simple_value=float(fruit_avg_value))

					self.summary_writer.add_summary(summary, fruits_analyzed)
					self.summary_writer.flush()

				fruits_analyzed += 1

			saver.save(sess, self.model_path+'/model-'+str(fruits_analyzed)+'.cptk')