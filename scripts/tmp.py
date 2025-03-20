import shutil
import os

in_dir = 'dataset/AnimalML3D/texts'

def get_files(directory):

    return [f for f in os.listdir(directory)]


dirs = get_files(in_dir)
print(len(dirs))
dirs = [f.replace('.txt', '') for f in dirs]

objj_dir = 'dataset/AnimalML3D/motions'
obj_dir = get_files(objj_dir)
print(len(obj_dir))
obj_dir = [f.replace('.bvh', '') for f in obj_dir]
num = 0
b = 0

for f in obj_dir:
    if f in dirs:
        num += 1

    else:
        b += 1
        print(f)

print(num)
print(b)



