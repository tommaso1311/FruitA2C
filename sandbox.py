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

		state = f.get_state(d)
		# action, action_idx = agent.policy()
		# value = agent.value()
		reward = f.add_guess(d, "same")
		print(reward)

	f.apply_UUID()

print([d for d in f.defects_analyzed])