from source.fruit.fruit import Fruit

load_path = "./dataset/sample/"

f = Fruit(0, load_path)

print(f.shots_keys)

ds = []

for d in f:
	ds.append(d)
	print(d)

print(ds)