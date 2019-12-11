from source.fruit.fruit import Fruit
from libxmp import XMPFiles, consts
import tifffile
import ast
from os import path

load_path = "./dataset/sample/"

f = Fruit(48, load_path)

print([d for s in f.shots_to_analyze for d in s.defects_to_analyze])