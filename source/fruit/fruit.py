from skimage.measure import label, regionprops
from source.fruit.shot import Shot
from libxmp import XMPFiles, consts
from source.fruit.defect import Defect
from uuid import uuid4
import numpy as np
import tifffile
import ast
from os import path

class Fruit:

	def __init__(self, fruit_ID, load_path, defects_thresholds=[160]):

		self.fruit_ID = fruit_ID

		self.shots_to_analyze = Fruit.load_shots(fruit_ID, load_path, defects_thresholds)
		self.shots_tot = len(self.shots_to_analyze)
		self.defects_tot = sum([shot.defects_tot for shot in self.shots_to_analyze])

		self.shots_analyzed = []
		self.current_shot = None

		self.is_analyzed = False
		self.update_current_shot()

	def __str__(self):
		return f"Fruit {self.fruit_ID}"

	def load_shots(fruit_ID, load_path, defects_thresholds):

		name = path.join(load_path, f"{fruit_ID}.tiff")
		shots_array = tifffile.TiffFile(name).asarray()
		xmpfile = XMPFiles(file_path=name, open_forupdate=True).get_xmp()
		defects_IDs_list = ast.literal_eval(xmpfile.get_property(consts.XMP_NS_DC, "description[1]"))

		shots = [Shot(i, shot_array, defects_IDs, defects_thresholds, fruit_ID)\
				for i, (shot_array, defects_IDs) in enumerate(zip(shots_array, defects_IDs_list))]

		return shots

	def update_current_shot(self):

		self.current_shot = self.shots_to_analyze.pop(0) if self.shots_to_analyze else None
		if not (self.shots_to_analyze or self.current_shot):
			self.is_analyzed = True

	def get_current_shot(self):

		current_shot = self.current_shot
		if current_shot.is_analyzed:
			self.shots_analyzed.append(self.current_shot)
			self.update_current_shot()

		return current_shot