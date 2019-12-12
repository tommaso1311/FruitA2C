from source.fruit.fruit import Fruit
from libxmp import XMPFiles, consts
import tifffile
import ast
from os import path

load_path = "./dataset/sample/"

f = Fruit(0, load_path)

while not f.is_analyzed and f.is_analyzable:
	print(f.get_current_defect())