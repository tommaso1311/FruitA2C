from source.fruit.fruit import Fruit
from libxmp import XMPFiles, consts
import tifffile
import ast
from os import path

load_path = "./dataset/sample/"

f = Fruit(0, load_path)
print([d for d in f.defects_analyzed])

while not f.is_analyzed and f.is_analyzable:
	
	# defect = f.get_current_defect()
	f.update_current_defect()

	for d in f.defects_analyzed:
		pass
		state = f.get_state(d)
		print(state)
		# action, action_idx = agent.policy()
		# value = agent.value()
		# reward = fruit.apply_action / add_guess
