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

		self.shots, self.defects_to_analyze = Fruit.load(fruit_ID, load_path, defects_thresholds)
		self.shots_tot = len(self.shots)
		self.defects_tot = sum([len(defects) for defects in self.defects_to_analyze])

		self.current_shot = None
		self.current_defect = None
		self.defects_analyzed = []
		self.is_analyzable = sum([shot.is_analyzable for shot in self.shots])>1
		self.is_analyzed = False

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

	def get_current_defect(self):

		self.current_shot = self.defects_to_analyze.pop(0) if (self.defects_to_analyze and not self.current_shot) else self.current_shot
		self.current_defect = self.current_shot.pop(0) if self.current_shot else self.current_defect
		self.is_analyzed = True if not self.defects_to_analyze else False

		return self.current_defect