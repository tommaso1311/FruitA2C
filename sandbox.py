# from source.fruit.fruit import Fruit
from fruit2 import Fruit
from libxmp import XMPFiles, consts
import tifffile
import ast
from os import path

load_path = "./dataset/sample/"

f = Fruit(0, load_path)

print(f.shots_analyzed)

while f.is_analyzable():
	f.update_current_defect()
	print(f.current_defect)
	# do smth
	f.apply_UUID()

print(f.shots_analyzed)
# while not f.is_analyzed and f.is_analyzable:
	
# 	f.update_current_defect()

# 	for d in f.defects_analyzed:
# 		if d.shot_number != f.current_defect.shot_number:
# 			print(f"Confronting {f.current_defect} with {d}")

# 			state = f.get_state(d)
# 			print(f"State is {state[0][1]}")
# 			# action, action_idx = agent.policy()
# 			# value = agent.value()
# 			reward = f.add_guess(d, "same")
# 			print(f"Reward is {reward}")
# 			print()

# 	f.apply_UUID()

# print([d for d in f.defects_analyzed])