import os
import numpy as np
import shutil
#from numpy.random import shuffle

dataset_path = "C:\\Users\\kzorina\\ONOVA\\FashionDataset\\Fashion10000\\Photos"
custom_catgories = ['Hoodie', 'Blouse','Dress','Jeans']
all_categories = sorted(os.listdir(dataset_path))
print(custom_catgories)


train_path = "train"
test_path = "test"
shutil.rmtree(train_path)
shutil.rmtree(test_path)
if not os.path.exists(train_path):
    os.makedirs(train_path)
else:
    filelist = [ f for f in os.listdir(train_path)]
    for f in filelist:
        os.remove(os.path.join(train_path, f))
if not os.path.exists(test_path):
    os.makedirs(test_path)
else:
    filelist = [ f for f in os.listdir(test_path)]
    for f in filelist:
        os.remove(os.path.join(test_path, f))

train_test_perc = 0.7
categories = custom_catgories
for i, category in enumerate(categories):
    os.makedirs(os.path.join(train_path, category))
    os.makedirs(os.path.join(test_path, category))

    category_path = os.path.join(dataset_path, category)
    files = []
    for file in os.listdir(category_path):
        path = os.path.join(category_path, file)
        files.append(file)
    ind = np.asarray(range(len(files)))
    np.random.shuffle(ind)
    train_size = int(len(files) * train_test_perc)
    train_ind = ind[:train_size]
    test_ind = ind[train_size:]
    for i in train_ind:
        old_path = os.path.join(category_path, files[i])
        path = os.path.join(train_path, os.path.join(category, files[i]))
        shutil.copyfile(old_path, path)
    for i in test_ind:
        old_path = os.path.join(category_path, files[i])
        path = os.path.join(test_path, os.path.join(category, files[i]))
        shutil.copyfile(old_path, path)


