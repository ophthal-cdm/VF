from detectron2.structures import BoxMode
from detectron2.data import build_detection_train_loader, build_detection_test_loader
from detectron2.data import transforms as T
from detectron2.data import detection_utils as utils
from detectron2.engine import DefaultTrainer
from detectron2.utils.visualizer import Visualizer, ColorMode

import torch
import cv2

import copy
import os
import xml.etree.ElementTree as elemTree

import math
import numpy as np
from pytesseract import image_to_string
from PIL import Image
import pandas
import csv

text_dir = '/home/ubuntu/data/Workspace/Soobin/wide_report/classes.txt'
# category 개수 count
def category_cnt(text_dir=text_dir):
    f = open(text_dir, "r")
    category_names = f.readlines()
    return len(category_names)
classes=category_cnt()

# xml file (annotation) 을 불러와 detectron2의 모델에 맞게 data 입력
def get_data_dicts(img_dir, annotation_dir, text_dir):
    dataset_dicts = []
    file_list = os.listdir(img_dir)
    idx = 0
    
    f = open(text_dir, "r")
    category_names = f.readlines()
    category_dict = {}
    # del category_names[0]
    category_idx = 0
    for category_name in category_names:
        category_dict[category_name.split("\n")[0]] = category_idx
        category_idx += 1

    for file_name in file_list:
        record = {}

        tree = elemTree.parse(os.path.join(annotation_dir, file_name.split(".")[0] + ".xml"))
        root = tree.getroot()

        width = int(root[4][0].text)
        height = int(root[4][1].text)
        idx += 1

        record["file_name"] = img_dir + file_name
        record["image_id"] = idx
        record["width"] = width
        record["height"] = height

        objects = []
        for i in range(len(root.findall("object"))):
            category = root.findall("object")[i].find("name").text
            category_idx = category_dict[category]
            xmin = int(root.findall("object")[i].find("bndbox").find("xmin").text)
            ymin = int(root.findall("object")[i].find("bndbox").find("ymin").text)
            xmax = int(root.findall("object")[i].find("bndbox").find("xmax").text)
            ymax = int(root.findall("object")[i].find("bndbox").find("ymax").text)

            obj = {
                "bbox":[xmin, ymin, xmax, ymax],
                "bbox_mode":BoxMode.XYXY_ABS,
                "category_id":category_idx,
                "iscrowd":0
            }
            objects.append(obj)
        record["annotations"] = objects
        dataset_dicts.append(record)
    
    return dataset_dicts


def get_predict_dicts(img_dir):
    dataset_dicts = []
    file_list = os.listdir(img_dir)
    idx = 0

    for file_name in file_list:
        record = {}

        height, width = cv2.imread(img_dir + file_name).shape[:2]
        idx += 1

        record["file_name"] = img_dir + file_name
        record["image_id"] = idx

        dataset_dicts.append(record)
    
    return dataset_dicts


# data annotation이 필요한 경우 transformation 함수를 사용하기 위해 설정해 놓음
def my_mapper(dataset_dict):
    dataset_dict = copy.deepcopy(dataset_dict)
    image = utils.read_image(dataset_dict["file_name"], format="BGR")
    image, transforms = T.apply_transform_gens([
        T.RandomExtent([0.8, 1.2], [1.5, 1.5]), 
    ], image)
    dataset_dict["image"] = torch.as_tensor(image.transpose(2,0,1).astype("float32"))

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image.shape[:2])
    dataset_dict["instances"] = utils.filter_empty_instances(instances)
    return dataset_dict


class My_Trainer(DefaultTrainer):
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=None)
    
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=my_mapper)


# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

#erosion
def erode(image):
    kernel = np.ones((3,3),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

def image_processing(img):
    gray = get_grayscale(img)
    # thresh = canny(gray)
    # clean = remove_noise(gray)
    # thred = thresholding(gray)
    # eroding = erode(gray)
    return gray


# 각 box들을 추출 후 저장
def get_output_image(predictor, image_dir, metadata, text_dir, expnum):
    image_list = os.listdir(image_dir)

    f = open(text_dir, "r")
    category_names = f.readlines()
    category_dict = {}
    # del category_names[0]
    category_idx = 0
    for category_name in category_names:
        category_dict[category_idx] = category_name.split("\n")[0]

        if not os.path.exists("./img%02i/category%02i_%s/"%(expnum, category_idx,category_dict[category_idx])):
            os.makedirs("./img%02i/category%02i_%s/"%(expnum, category_idx,category_dict[category_idx]))
       
        category_idx += 1

    for image_name in image_list:
        image = cv2.imread(image_dir + image_name)
        outputs = predictor(image)

        v = Visualizer(image[:,:,::-1],
                    metadata=metadata,
                    scale=1.0,
                    instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        cv2.imwrite("./img%02i/%s"%(expnum, image_name), v.get_image()[:,:,::-1])


        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()

        for i in range(len(boxes)):
            pred_box = boxes[i]
            pred_class = classes[i]
            # if category_dict[pred_class]=='erase':
            #     erase = image.copy()
            #     erase[int(pred_box[1]):int(pred_box[3]), int(pred_box[0]):int(pred_box[2])] = (255,255,255)
            #     cv2.imwrite("./img%02i/category%02i_%s/%s"%(expnum, pred_class, category_dict[pred_class], image_name), erase)
            #     continue
            cropped_img = image[int(pred_box[1]):int(pred_box[3]), int(pred_box[0]):int(pred_box[2])]
            saving_img = image_processing(cropped_img)
            cv2.imwrite("./img%02i/category%02i_%s/%s"%(expnum, pred_class, category_dict[pred_class], image_name), saving_img)
           

def rename_file(predictor, image_dir, metadata, text_dir):
    image_list = os.listdir(image_dir)
    f = open(text_dir, "r")
    category_names = f.readlines()
    category_dict = {}
    # del category_names[0]
    category_idx = 0
    for category_name in category_names:
        category_dict[category_idx] = category_name.split("\n")[0]
        category_idx += 1

    for image_name in image_list:
        image = cv2.imread(image_dir + image_name)
        outputs = predictor(image)

        v = Visualizer(image[:,:,::-1],
                    metadata=metadata,
                    scale=1.0,
                    instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))

        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        
        for i in range(len(boxes)):
            pred_box = boxes[i]
            pred_class = classes[i]
            if category_dict[pred_class]=='side':
                cropped = image[int(pred_box[1]):int(pred_box[3]), int(pred_box[0]):int(pred_box[2])]
                text = image_to_string(cropped,config="-c tessedit_char_whitelist=(ODSLR) -psm 6")
                print(text)

                if 'R' in text:
                    os.rename(image_dir+image_name,image_dir+image_name[:-4]+'_dl_OD.jpg')
                    print(image_dir+image_name,image_dir+image_name[:-4]+'_dl_OD.jpg')
                elif 'L' in text:
                    os.rename(image_dir+image_name,image_dir+image_name[:-4]+'_dl_OS.jpg')
                    print(image_dir+image_name,image_dir+image_name[:-4]+'_dl_OS.jpg')
                else:
                    continue
                

def del_custom_id_rename(predictor, image_dir, metadata, expnum, eye_dict):
    image_list = os.listdir(image_dir)

    if not os.path.exists("./img%02i/del_custom_id/"%(expnum)):
        os.mkdir("./img%02i/del_custom_id/"%(expnum))
    
    for image_name in image_list:
        image = cv2.imread(image_dir + image_name)
        outputs = predictor(image)

        v = Visualizer(image[:,:,::-1],
                    metadata=metadata,
                    scale=1.0,
                    instance_mode=ColorMode.IMAGE)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))


        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        classes = outputs["instances"].pred_classes.cpu().numpy()
        
        eye = eye_dict[image_name]

        if eye == 0:
            eye = "OD"  # 오른쪽
        elif eye == 1:
            eye = "OS"  # 왼  쪽

        box_dict = {}
        for index in range(len(boxes)):
            box_dict[classes[index]] = boxes[index]


        for i in [0, 1, 3, 15, 25]:
            if i in box_dict.keys():
                pred_box = box_dict[i]

                image[int(pred_box[1]):int(pred_box[3]), int(pred_box[0]):int(pred_box[2])] = image[1,1]
                cv2.imwrite("./img%02i/del_custom_id/%s_dl_%s.jpg"%(expnum, image_name.split(".")[0],eye), image)


def read_OCT_master_table():
    table_dir='/home/ubuntu/data/Workspace/Soobin/data/frequent_list/OCT var master tableV2.csv'
    master_table = pandas.read_csv(table_dir)
    return master_table



def num_extract(path,img):
    width = 60
    height=30
    top = int(height*0.15)
    bottom=top
    left = int(width*0.15)
    right=left
    dim = (width,height)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,dim,interpolation=cv2.INTER_NEAREST)
    blur = cv2.GaussianBlur(img,(3,3),0)
    ret, temp = cv2.threshold(blur, int(cv2.mean(blur)[0]), 255,cv2.THRESH_TRUNC)
    ret, thr2 = cv2.threshold(temp, int(cv2.mean(temp)[0]), 255, cv2.THRESH_BINARY)
    im2 = cv2.copyMakeBorder(thr2,top,bottom,left,right,cv2.BORDER_CONSTANT,None,[255,255,255])
    # cv2.imwrite(path+'_processed.jpg',im2)
    text = image_to_string(im2, config="-c tessedit_char_whitelist=0123456789 -psm 6")
    return text


master_table = read_OCT_master_table()
def add_csv():
    mat = master_table
    

def extract_txt_add_csv(image_dir, text_dir, expnum):
    master_table = read_OCT_master_table()
    
    image_list = os.listdir(image_dir)
    f = open(text_dir, "r")
    category_names = f.readlines()
    category_dict = {}
    category_list=[]
  
    category_idx = 0
    for category_name in category_names:
        category_dict[category_idx] = category_name.split("\n")[0]
        category_idx += 1

    for category_idx in range(0,classes):
        category_list.append(category_dict[category_idx])

    for image_name in image_list:
        for category_idx in range(0,classes):
            image_file = "./img%02i/category%02i_%s/%s"%(expnum, category_idx, category_dict[category_idx], image_name)
            if os.path.exists(image_file):
                txt_file = open("./img%02i/category%02i_%s/%s.txt"%(expnum, category_idx, category_dict[category_idx], image_name),'w')
                image = cv2.imread(image_file)
                if category_idx in [4,5,6,7,8,9,10,11,12]:    
                    # text = num_preprocess_extract(image_file)        # dpi upgraded
                    text = num_extract(image_file,image)
                elif category_idx in [13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,37]:
                    text = image_to_string(image, config="-c tessedit_char_whitelist=0123456789 -psm 6")
                else:
                    text = image_to_string(image)
                
                    # if 'R' is in text:
                    #     rename_official
                txt_file.write(text)
                txt_file.close()
        



def rename_official(image_dir,expnum):
    for image_name in image_list:
        csv_save_list = []
        txt_file = open("./img%02i/text/%s.txt"%(expnum, image_name.split(".")[0]), 'w')
        csv_save_list.append(image_name.split(".")[0])
        for category_idx in range(0,classes):
            image_file = "./img%02i/category%02i_%s/%s"%(expnum, category_idx, category_dict[category_idx], image_name)
            if os.path.exists(image_file):
                image = cv2.imread(image_file)
                resize_image = cv2.resize(image, dsize=(0,0), fx=4.3, fy=4.3, interpolation=cv2.INTER_LANCZOS4)
                text = image_to_string(resize_image)
                # csv_save_list.append(text)
                txt_file.write(text)
                
            else:
                continue

            if category_idx == 2:
                if "R" in text:
                    eye_dict[image_name] = 0
                else:
                    eye_dict[image_name] = 1



# 추출한 text data를 .txt file로 저장
def generate_txt_data(image_dir, text_dir, expnum, times):
    f = open(text_dir, "r")
    category_names = f.readlines()
    category_list=["file_name"]
    category_dict = {}
    # del category_names[0]
    category_idx = 0
    for category_name in category_names:
        category_dict[category_idx] = category_name.split("\n")[0]
        category_idx += 1

    for category_idx in range(0,classes):
        category_list.append(category_dict[category_idx])
    

    if not os.path.exists("./img%02i/text/"%(expnum)):
        os.mkdir("./img%02i/text/"%(expnum))
    
    image_list = os.listdir(image_dir)

    eye_dict = {}

    csv_total_list =[]

    for image_name in image_list:
        csv_save_list = []
        txt_file = open("./img%02i/text/%s.txt"%(expnum, image_name.split(".")[0]), 'w')
        csv_save_list.append(image_name.split(".")[0])
        for category_idx in range(0,classes):
            image_file = "./img%02i/category%02i_%s/%s"%(expnum, category_idx, category_dict[category_idx], image_name)
            if os.path.exists(image_file):
                image = cv2.imread(image_file)
                resize_image = cv2.resize(image, dsize=(0,0), fx=4.3, fy=4.3, interpolation=cv2.INTER_LANCZOS4)
                text = image_to_string(resize_image)
                # csv_save_list.append(text)
                txt_file.write(text)
                
            else:
                continue

            if category_idx == 2:
                if "R" in text:
                    eye_dict[image_name] = 0
                else:
                    eye_dict[image_name] = 1

        csv_total_list.append(csv_save_list)
    
    
    # csv_file = pandas.DataFrame(csv_total_list, columns=category_list)
    # csv_file.to_csv("./img%02i/VF_text_result.csv"%(expnum), index=False, encoding='utf-8')

        # txt_file.close()

    return eye_dict
