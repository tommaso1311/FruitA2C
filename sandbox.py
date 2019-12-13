from source.fruit.fruit import Fruit

load_path = "./dataset/sample/"

f = Fruit(0, load_path)

while f.is_analyzable():
	f.update_current_defect()

	for d in f.get_defects_analyzed():
		print(f"Confronting {f.current_defect} with {d}")

		state = f.get_state(d)
		print(f"State is {state[0]}")

		# action step
		# value step

		reward = f.add_guess(d, "same")
		print(f"Reward is {reward}")
		print()

	f.apply_UUID()