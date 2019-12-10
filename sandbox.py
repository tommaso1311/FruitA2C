from source.fruit.fruit import Fruit

load_path = "./dataset/sample/"

f = Fruit(0, load_path)

print(f.shots_keys)

ds = []

for d in f:
	ds.append(d)

print(ds[0]-ds[1])