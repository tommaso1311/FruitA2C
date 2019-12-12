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

		self.shots, self.defects = Fruit.load(fruit_ID, load_path, defects_thresholds)

		# self.defects_analyzed = []
		# self.defects_tot = sum([shot.defects_tot for shot in self.shots])

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

	def load_defects(shot_number, img_array, defects_IDs, defects_thresholds):

		thresholded_img = img_array < defects_thresholds[0]
		labels = label(thresholded_img)
		properties = regionprops(labels, coordinates="rc")

		defects = [Defect(defect_ID, props, shot_number, img_array.shape)\
					for defect_ID, props in zip(defects_IDs, properties)]

		return defects

	def load(fruit_ID, load_path, defects_thresholds):

		shots = Fruit.load_shots(fruit_ID, load_path, defects_thresholds)

		defects = []
		for shot in shots:
			defects.extend(shot.defects)

		return shots, defects