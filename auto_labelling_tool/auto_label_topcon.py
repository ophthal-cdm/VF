import glob
import cv2
import os
import xml.etree.ElementTree as ET
import shutil

path="/home/ubuntu/data/Workspace/Soobin/wide_report/annotation/"
# for f_path in glob.glob(path+"*"):
#     im = cv2.imread(f_path)
#     # print(im.shape[:2])
#     if not os.path.isfile(path)

# src = '/home/ubuntu/data/Workspace/Soobin/data/frequent_list/topcon_macular_labeled/annotation/FEZ79605_384640145_20161101_20161101_21213467_001.xml'
# src = ET.parse(src)
# root = src.getroot()
# element = root[2]
# print(element)
# ori = element.text
# print(ori[:68])
# element.text= 'hi'
# print(element.text)

def edit_element():
    for i,f in enumerate(os.listdir(path)):
        src = ET.parse(path+f)
        root = src.getroot()
        edit = 'side'
        element = root[6]       # root태그 안에 속하는 element tag를 의미
        element.find('name').text = edit 
        src.write(path+f)
    print('done')

    

def auto_labelling(path):
    for i,f in enumerate(os.listdir(path+'data/')):
        xml = f[:-4]+".xml"
        src = '/home/ubuntu/data/Workspace/Soobin/wide_report/annotation/FEZ79605_1116434616_20171012_20171012_28907265_001.xml'
        src = ET.parse(src)
        print(src)
        if not os.path.isfile(path+'annotation/'+f[:-4]+'.xml'):
            root = src.getroot()
            element = root[1]           
            element.text= f[:-4]+".JPG"    # change <file> to the file name
            print(element.text)
            p = root[2]
            ori = p.text
            p.text = ori[:68]+f[:-4]+'.JPG'
            src.write('/home/ubuntu/data/Workspace/Soobin/wide_report/annotation/'+f[:-4]+'.xml')
    print("done")

auto_labelling('/home/ubuntu/data/Workspace/Soobin/wide_report/')
