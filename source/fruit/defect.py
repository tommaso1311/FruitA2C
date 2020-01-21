import numpy as np
from math import pi as PI

class Defect:

	def __init__(self, ID, props, shot_number, shot_size=None, props_ready=False):

		self._ID = ID
		self.shot_number = shot_number

		if not props_ready:
			self._props = Defect.load_props(props, shot_size)
		else:
			self._props = props
		self.guesses = []

		self.UUID = None
		self.is_analyzed = False

	def __repr__(self):
		return f"Defect {self._ID} of Shot {self.shot_number} with UUID: {self.UUID}"

	def __str__(self):
		return f"Defect {self._ID} of Shot {self.shot_number} with UUID: {self.UUID}"

	def __eq__(self, defect):
		return self._ID == defect._ID

	def __sub__(self, defect):

		delta = np.zeros(len(self._props))

		for i, (key, val) in enumerate(self._props.items()):
			delta[i] = 1 - np.abs(val - defect._props[key])

		return np.array(delta).reshape((1, len(self._props)))

	def load_props(props, shot_size):

		props_dict = {}

		# props_dict["area"] = props.area
		# props_dict["perimeter"] = props.perimeter
		props_dict["center_x"] = props.centroid[0]/shot_size[1]
		props_dict["center_y"] = props.centroid[1]/shot_size[0]
		props_dict["eccentricity"] = props.eccentricity
		props_dict["solidity"] = props.solidity
		props_dict["circularity"] = (4*PI*props.area)/(props.perimeter**2) if (props.perimeter**2) else 1

		return props_dict