from libxmp import XMPFiles, consts
from uuid import uuid4
import tifffile
import ast
import numpy as np
from os import path
from source.a2c import utils
from skimage.measure import label, regionprops
from defect2 import Defect

class Fruit:

	def __init__(self, fruit_ID, load_path, defects_thresholds=[160]):
		
		self._fruit_ID = fruit_ID

		self.shots = Fruit.load_shots(fruit_ID, load_path, defects_thresholds)

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