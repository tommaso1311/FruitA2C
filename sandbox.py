from source.fruit.fruit import Fruit
from source.fruit.shot import Shot
from skimage.measure import label, regionprops
from libxmp import XMPFiles, consts
from source.fruit.defect import Defect
from uuid import uuid4
import numpy as np
import tifffile
import ast
from os import path

load_path = "./dataset/sample/"
name = path.join(load_path, "0.tiff")
shots = tifffile.TiffFile(name).asarray()
xmpfile = XMPFiles(file_path=name, open_forupdate=True).get_xmp()
answers = ast.literal_eval(xmpfile.get_property(consts.XMP_NS_DC, "description[1]"))

index = 0

shot = shots[index, :, :]
s = Shot(index, shot, answers[index], [160], 0)

print(s)
print(s.defects)

# f = Fruit(0, load_path)

# print(f.shots_keys)

# ds = []

# for d in f:
# 	ds.append(d)
# 	print(d)

# print(ds)