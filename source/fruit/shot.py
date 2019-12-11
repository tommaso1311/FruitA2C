from skimage.measure import label, regionprops
from source.fruit.defect import Defect

class Shot:

	def __init__(self, shot_ID, img_array, defects_IDs, defects_thresholds, fruit_ID):

		self.shot_ID = shot_ID
		self.fruit_ID = fruit_ID

		self.defects = Shot.load_defects(shot_ID, img_array, defects_IDs, defects_thresholds)

	def __str__(self):
		return f"Shot {self.shot_ID} of Fruit {self.fruit_ID}"

	def load_defects(shot_ID, img_array, defects_IDs, defects_thresholds):

		thresholded_img = img_array < defects_thresholds[0]
		labels = label(thresholded_img)
		properties = regionprops(labels, coordinates="rc")

		defects = [Defect(defect_ID, props, shot_ID, img_array.shape)\
					for defect_ID, props in zip(defects_IDs, properties)]

		return defects