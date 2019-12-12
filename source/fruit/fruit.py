from source.fruit.shot import Shot
from libxmp import XMPFiles, consts
from uuid import uuid4
import tifffile
import ast
import numpy as np
from os import path
from source.a2c import utils

class Fruit:

	def __init__(self, fruit_ID, load_path, defects_thresholds=[160]):

		self.fruit_ID = fruit_ID

		self.shots, self.defects_to_analyze = Fruit.load(fruit_ID, load_path, defects_thresholds)
		self.shots_tot = len(self.shots)
		self.defects_tot = sum([len(defects) for defects in self.defects_to_analyze])

		self.current_shot = None
		self.current_defect = None

		self.defects_analyzed = []
		self.is_analyzable = sum([shot.is_analyzable for shot in self.shots])>1
		self.is_analyzed = False

		self.set_starting_UUIDs()

	def __str__(self):
		return f"Fruit {self.fruit_ID}"

	def load_shots(fruit_ID, load_path, defects_thresholds):

		name = path.join(load_path, f"{fruit_ID}.tiff")
		shots_array = tifffile.TiffFile(name).asarray()
		xmpfile = XMPFiles(file_path=name).get_xmp()
		defects_IDs_list = ast.literal_eval(xmpfile.get_property(consts.XMP_NS_DC, "description[1]"))

		shots = [Shot(i, shot_array, defects_IDs, defects_thresholds, fruit_ID)\
				for i, (shot_array, defects_IDs) in enumerate(zip(shots_array, defects_IDs_list))]

		return shots

	def load(fruit_ID, load_path, defects_thresholds):

		shots = Fruit.load_shots(fruit_ID, load_path, defects_thresholds)

		defects = []
		for shot in shots:
			if shot.defects:
				defects.append(shot.defects)

		return shots, defects

	def update_current_defect(self):

		self.current_shot = self.defects_to_analyze.pop(0) if (self.defects_to_analyze and not self.current_shot) else self.current_shot
		self.current_defect = self.current_shot.pop(0) if self.current_shot else self.current_defect
		self.is_analyzed = not (self.current_shot or self.defects_to_analyze)

	def set_starting_UUIDs(self):
		self.current_shot = self.defects_to_analyze.pop(0) if (self.defects_to_analyze and not self.current_shot) else self.current_shot
		for _ in range(len(self.current_shot)):
			self.update_current_defect()
			self.current_defect.UUID = uuid4()
			self.defects_analyzed.append(self.current_defect)

	def get_rolling_state(self):

		shots_progress = self.current_defect.shot_number/self.shots_tot
		defects_progress = len(self.defects_analyzed)/self.defects_tot

		rolling_state = np.array([shots_progress, defects_progress]).reshape((1, 2))

		return rolling_state

	def get_state(self, defect):

		rolling_state = self.get_rolling_state()
		delta_state = self.current_defect - defect

		return np.hstack((rolling_state, delta_state))

	def add_guess(self, defect, action):

		print(action)
		if action == utils.consts.SAME:
			self.current_defect.guesses.append(defect.UUID)
			if self.current_defect == defect:
				return utils.consts.CORRECT_GUESS_REWARD
			else:
				return utils.consts.WRONG_GUESS_REWARD
		elif self.action == utils.consts.DIFFERENT:
			if self.current_defect == defect:
				return utils.consts.WRONG_GUESS_REWARD
			else:
				return utils.consts.CORRECT_GUESS_REWARD

	def apply_UUID(self):

		new_UUID = max(set(self.current_defect.guesses), key=self.current_defect.guesses.count)
		self.current_defect.UUID = new_UUID
		self.defects_analyzed.append(self.current_defect)