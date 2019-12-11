from skimage.measure import label, regionprops
from source.fruit.defect import Defect

class Shot:

	def __init__(self, shot_ID, img_array, defects_IDs, defects_thresholds, fruit_ID):

		self.shot_ID = shot_ID
		self.fruit_ID = fruit_ID

		self.defects_to_analyze = Shot.load_defects(shot_ID, img_array, defects_IDs, defects_thresholds)
		self.defects_tot = len(self.defects_to_analyze)

		self.defects_analyzed = []
		self.current_defect = None
		
		self.is_analyzed = False
		self.update_current_defect()

	def __str__(self):
		return f"Shot {self.shot_ID} of Fruit {self.fruit_ID}"

	def __repr__(self):
		return f"Shot {self.shot_ID} of Fruit {self.fruit_ID}"

	def load_defects(shot_ID, img_array, defects_IDs, defects_thresholds):

		thresholded_img = img_array < defects_thresholds[0]
		labels = label(thresholded_img)
		properties = regionprops(labels, coordinates="rc")

		defects = [Defect(defect_ID, props, shot_ID, img_array.shape)\
					for defect_ID, props in zip(defects_IDs, properties)]

		return defects

	def update_current_defect(self):

		self.current_defect = self.defects_to_analyze.pop(0) if self.defects_to_analyze else None
		if not (self.defects_to_analyze or self.current_defect):
			self.is_analyzed = True

	def get_current_defect(self):

		current_defect = self.current_defect
		self.defects_analyzed.append(self.current_defect)
		self.update_current_defect()

		return current_defect