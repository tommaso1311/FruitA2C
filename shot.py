class Shot:

	def __init__(self, ID):

		self.ID = ID
		self.defects = []

	def append(self, defect):
		self.defects.append(defect)