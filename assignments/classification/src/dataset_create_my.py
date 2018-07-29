import os
import numpy as np
import glob
import fnmatch

import PIL
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sys
import logging

fashion_dataset_path='C:/Users/kzorina/ONOVA/deepfashiondata/'

dataset_path='C:/Users/kzorina/Studing/CV/6DLinCV/ucu2018_dl4cv/assignments/classification/data/deepfashion'
dataset_train_path=os.path.join(dataset_path, 'train')
dataset_val_path=os.path.join(dataset_path, 'validation')
dataset_test_path=os.path.join(dataset_path, 'test')

category_name_generate=['Tee', 'Hoodie', 'Skirt', 'Shorts', 'Dress', 'Jeans']
category_names = []
with open(fashion_dataset_path + '/Anno/list_category_cloth.txt') as file_list_category_cloth:
        next(file_list_category_cloth)
        next(file_list_category_cloth)
        for line in file_list_category_cloth:
            word=line.strip()[:-1].strip().replace(' ', '_')
            category_names.append(word)


def create_dataset_split_structure():

    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    if not os.path.exists(dataset_train_path):
        os.makedirs(dataset_train_path)

    if not os.path.exists(dataset_val_path):
        os.makedirs(dataset_val_path)

    if not os.path.exists(dataset_test_path):
        os.makedirs(dataset_test_path)

def create_category_structure():
	for category_name in category_name_generate:
		# Train
		category_path_name=os.path.join(dataset_train_path, category_name)
		logging.debug('category_path_name {}'.format(category_path_name))
		if not os.path.exists(os.path.join(category_path_name)):
			os.makedirs(category_path_name)

		# Validation
		category_path_name=os.path.join(dataset_val_path, category_name)
		logging.debug('category_path_name {}'.format(category_path_name))
		if not os.path.exists(os.path.join(category_path_name)):
			os.makedirs(category_path_name)

		# Test
		category_path_name=os.path.join(dataset_test_path, category_name)
		logging.debug('category_path_name {}'.format(category_path_name))
		if not os.path.exists(os.path.join(category_path_name)):
			os.makedirs(category_path_name)


def get_dataset_split_name(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            dataset_split_name=line.split()[1]
            #logging.debug('dataset_split_name {}'.format(dataset_split_name))
            return dataset_split_name.strip()

def get_gt_bbox(image_path_name, file_ptr):
    for line in file_ptr:
        if image_path_name in line:
            x1=int(line.split()[1])
            y1=int(line.split()[2])
            x2=int(line.split()[3])
            y2=int(line.split()[4])
            bbox = [x1, y1, x2, y2]
            logging.debug('bbox {}'.format(bbox))
            return bbox

def calculate_bbox_score_and_save_img(image_path_name, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2):

    try:
        img_read = Image.open(image_path_name)
        dataset_image_path = dataset_image_path.replace("\\","/")
        # Ground Truth
        image_save_name = image_path_name.split('/')[-2] + '_' + image_path_name.split('/')[-1].split('.')[0]
        image_save_path = dataset_image_path.rsplit('/', 1)[0]
        image_save_path_name = image_save_path + '/' + image_save_name + '_gt_' +  str(gt_x1) + '-' + str(gt_y1) + '-' + str(gt_x2) + '-' + str(gt_y2) + '_iou_' +  '1.0' + '.jpg'
        logging.debug('image_save_path_name {}'.format(image_save_path_name))
        img_crop = img_read.crop((gt_x1, gt_y1, gt_x2, gt_y2))
        img_crop.save(image_save_path_name)
        #img_crop.save(image_save_path_name)
        #logging.debug('img_crop {} {} {}'.format(img_read.format, img_read.size, img_read.mode))
    except FileNotFoundError:
        print("File {} does not exist".format(image_path_name))
    except:
        print("Error: {}".format(sys.exc_info()[0]))

			
def generate_dataset_images():
    # 1600 train, 400 test per category
    count = np.zeros((3,len(category_name_generate)))
    with open(fashion_dataset_path + '/Anno/list_bbox.txt') as file_list_bbox_ptr:
        with open(fashion_dataset_path + '/Anno/list_category_img.txt') as file_list_category_img:
            with open(fashion_dataset_path + '/Eval/list_eval_partition.txt', 'r') as file_list_eval_ptr:
                next(file_list_category_img)
                next(file_list_category_img)
                for line in file_list_category_img:

                    line = line.split()
                    image_path_name = line[0]
                    image_name = line[0].split('/')[-1]
                    image_full_name = line[0].replace('/', '_')
                    image_category_index=int(line[1:][0]) - 1

                    dataset_image_path = ''
                    dataset_split_name = get_dataset_split_name(image_path_name, file_list_eval_ptr)

                    if dataset_split_name == "train":
                        '''try:
                            count[0][category_name_generate.index(category_names[image_category_index])] += 1
                        except:
                            pass'''
                        dataset_image_path = os.path.join(dataset_train_path, category_names[image_category_index], image_full_name)
                    elif dataset_split_name == "val":
                        '''try:
                            count[2][category_name_generate.index(category_names[image_category_index])] += 1
                        except:
                            pass'''
                        dataset_image_path = os.path.join(dataset_val_path, category_names[image_category_index], image_full_name)
                    elif dataset_split_name == "test":
                        '''try:
                            count[1][category_name_generate.index(category_names[image_category_index])] += 1
                        except:
                            pass'''
                        dataset_image_path = os.path.join(dataset_test_path, category_names[image_category_index], image_full_name)
                    else:
                        logging.error('Unknown dataset_split_name {}'.format(dataset_image_path))
                        exit(1)

                     # Get ground-truth bounding boxes
                    gt_x1, gt_y1, gt_x2, gt_y2 = get_gt_bbox(image_path_name, file_list_bbox_ptr)                              # Origin is top left, x1 is distance from y axis;

                    image_path_name_src = os.path.join(fashion_dataset_path, 'Img', image_path_name)

                    calculate_bbox_score_and_save_img(image_path_name_src, dataset_image_path, gt_x1, gt_y1, gt_x2,
                                                      gt_y2)
                    '''
                    try:
                        if (dataset_split_name == "train") and (count[0][category_name_generate.index(category_names[image_category_index])]<1601):
                            calculate_bbox_score_and_save_img(image_path_name_src, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2)
                        if (dataset_split_name == "test") and (count[1][category_name_generate.index(category_names[image_category_index])] < 401):
                            calculate_bbox_score_and_save_img(image_path_name_src, dataset_image_path, gt_x1, gt_y1, gt_x2, gt_y2)
                    except:
                        pass
                        '''



                    #logging.info('count {} {}'.format(count, dataset_image_path))
						
				
					
if __name__ == '__main__':

    create_dataset_split_structure()
    create_category_structure()
    generate_dataset_images()
    #display_category_data()