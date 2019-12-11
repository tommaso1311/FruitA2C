from source.fruit.fruit import Fruit
from libxmp import XMPFiles, consts
import tifffile
import ast
from os import path

load_path = "./dataset/sample/"

f = Fruit(12, load_path)

while not f.is_analyzed:
	s = f.get_current_shot()
	while not s.is_analyzed:
		d = s.get_current_defect()
		
		print(d)
		print(f.shots_analyzed)
		print(f.current_shot)
		print(f.shots_to_analyze)
		print()