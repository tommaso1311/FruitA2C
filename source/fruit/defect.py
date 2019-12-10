import numpy as np
import math
# from src.fruit.utils import log10_trasform, sigmoid
# from uuid import uuid4

class Defect:

	def __init__(self, ID, props, shot_ID, shot_size):
		"""
		Instantiates Defect objects

		Parameters
		----------
		ID : int
			ID of the defect (on the fruit)
		props : RegionProps object
			properties of the region
		shot_size : array
			sizes of the shot
		"""

		self.ID = ID
		self.shot_ID = shot_ID
		self.UUID = None

		self.props = Defect.load_props(props, shot_size)
		self.guesses = []

	def __repr__(self):
		return f"Defect {self.ID} of Shot {self.shot_ID}"

	def __str__(self):
		return f"Defect {self.ID} of Shot {self.shot_ID} with UUID: {self.UUID}"

	def __eq__(self, defect):
		"""
		Used to make a comparison between two defects
		"""

		return True if self.ID == defect.ID else False

	def __sub__(self, defect):
		"""
		Used to return differences between two defects, in absolute value.
		Final value is between 0 (different) and 1 (same)
		"""

		len_props = len(self.props)
		delta = np.zeros(len_props)

		for i, (key, val) in enumerate(self.props.items()):
			delta[i] = 1 - np.abs(val - defect.props[key])

		return np.array(delta).reshape((1, len_props))

	def load_props(props, shot_size):
		"""
		Used to load properties inside a dict()
		"""

		props_dict = {}

		# props_dict["area"] = props.area
		# props_dict["perimeter"] = props.perimeter
		props_dict["center_x"] = props.centroid[0]/shot_size[1]
		props_dict["center_y"] = props.centroid[1]/shot_size[0]
		props_dict["eccentricity"] = props.eccentricity
		props_dict["solidity"] = props.solidity
		props_dict["circularity"] = (4*math.pi*props.area)/(props.perimeter**2)

		return props_dict