from libxmp import XMPFiles, consts
from uuid import uuid4
import tifffile
import ast
import numpy as np
from os import path
from source.a2c import utils
from skimage.measure import label, regionprops
from source.fruit.defect import Defect

class Fruit:

	def __init__(self, fruit_ID, load_path, defects_thresholds=[160]):
		
		self._fruit_ID = fruit_ID

		self.shots = Fruit.load_shots(fruit_ID, load_path, defects_thresholds)

		self._shots_tot = len(self.shots)
		self._defects_tot = sum([len(shot) for shot in self.shots])

		self._current_shot = []
		self.current_defect = None

		self.shots_analyzed = []
		self.set_starting_UUIDs()

	def __str__(self):
		return f"Fruit {self._fruit_ID}"

	def load_defects(shot_number, shot_array, defects_IDs, defects_thresholds):

		thresholded_img = shot_array < defects_thresholds[0]
		labels = label(thresholded_img)
		properties = regionprops(labels, coordinates="rc")

		defects = [Defect(defect_ID, props, shot_number, shot_array.shape)\
					for defect_ID, props in zip(defects_IDs, properties)]

		return defects

	def load_shots(fruit_ID, load_path, defects_thresholds):

		name = path.join(load_path, f"{fruit_ID}.tiff")
		shots_array = tifffile.TiffFile(name).asarray()
		xmpfile = XMPFiles(file_path=name).get_xmp()
		defects_IDs_list = ast.literal_eval(xmpfile.get_property(consts.XMP_NS_DC, "description[1]"))

		fruit_shots = []
		for shot_number, (shot_array, defects_IDs) in enumerate(zip(shots_array, defects_IDs_list)):
			shot_defects = Fruit.load_defects(shot_number, shot_array, defects_IDs, defects_thresholds)
			fruit_shots.append(shot_defects)

		return fruit_shots

	def defects_to_analyze(self):
		return any([not defect.is_analyzed for defect in self._current_shot])

	def shots_to_analyze(self):
		return any(self.shots)

	def is_analyzable(self):
		return self.defects_to_analyze() or self.shots_to_analyze()

	def update_current_shot(self):
		if all([defect.is_analyzed for defect in self._current_shot]):
			next_shot_index = next(i for i, shot in enumerate(self.shots) if shot) if self.is_analyzable() else 0
			self._current_shot = self.shots.pop(next_shot_index)

	def update_current_defect(self):
		self.update_current_shot()

		if not self.current_defect or self.current_defect.is_analyzed:
			next_defect_index = next(i for i, defect in enumerate(self._current_shot) if not defect.is_analyzed)
			self.current_defect = self._current_shot[next_defect_index]

	def apply_UUID(self):

		if self.current_defect.guesses:
			new_UUID = max(set(self.current_defect.guesses), key=self.current_defect.guesses.count)
		else:
			new_UUID = uuid4()

		self.current_defect.UUID = new_UUID

		self.current_defect.is_analyzed = True
		if not self.defects_to_analyze():
			self.shots_analyzed.append(self._current_shot)

	def set_starting_UUIDs(self):

		self.update_current_shot()
		for _ in self._current_shot:
			self.update_current_defect()
			self.apply_UUID()

	def get_defects_analyzed(self):

		defects_analyzed = [d for s in self.shots_analyzed for d in s]
		return defects_analyzed

	def get_rolling_state(self):

		shots_progress = self.current_defect.shot_number/self._shots_tot
		defects_progress = sum([len(shot) for shot in self.shots_analyzed])/self._defects_tot

		rolling_state = np.array([shots_progress, defects_progress]).reshape((1, 2))
		return rolling_state

	def get_state(self, defect):

		rolling_state = self.get_rolling_state()
		delta_state = self.current_defect - defect

		return np.hstack((rolling_state, delta_state))

	def add_guess(self, defect, action):

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