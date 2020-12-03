# import detectron2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import detecron2 library
from detectron2 import model_zoo
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.visualizer import Visualizer, ColorMode

from detectron2.checkpoint import DetectionCheckpointer

# personal library
from dict_with import get_data_dicts, get_predict_dicts, My_Trainer, get_output_image, extract_txt_add_csv, rename_file

# 기타 library
import cv2
import argparse
import os
import time
import torch

torch.backends.cudnn.benchmark=True
# hyper parameter 선언
parser = argparse.ArgumentParser(description="Train by detectron2 to detect custom dataset")
parser.add_argument("--gpu",
                    default="1", required=False, type=str, help="GPU Number")
parser.add_argument("--lr",
                    default=0.001, required=False, type=float, help="Learning Rate")
parser.add_argument("--batch",
                    default=4, required=False, type=int, help="Batch Size")
parser.add_argument("--iter",
                    default=20000, required=False, type=int, help="Iteration Number")
parser.add_argument("--image_path", default="/home/ubuntu/data/Workspace/Soobin/wide_report/",
                    required=False, type=str, help="Train Folder")
parser.add_argument("--image_name",
                    default="oct", required=False, type=str, help="Kind of Training Set")
parser.add_argument("--expnum",
                    default="88", required=False, type=int, help="Experiment Number: Separating each Experiment")

args = parser.parse_args()


# gpu number
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
print("The number of using GPU is %s" %(args.gpu))

# Direction
img_dir = args.image_path+'data/'                # Image File dir (.jpg)
annotation_dir = args.image_path+'annotation/'          # Annotation File dir (.xml)
text_dir = args.image_path + "classes.txt"                     # Text File dir (.txt)

# Test Image File dir
test_dir = '/home/ubuntu/data/Workspace/Soobin/data/detectron/unidentified_higher_resolution/category00_erase/'
# test_dir = '/home/ubuntu/data/Workspace/Soobin/data/detectron/unidentified_higher_resolution/category00_erase/'
# Loading Text File to generate category
f = open(text_dir, "r")
category_names = f.readlines()
category_list = []
for category_name in category_names:
    category_list.append(category_name.split("\n")[0])
print(category_list)

# Prepare Training Set
for d in [""]:
    DatasetCatalog .register(args.image_name+'wide_train/', lambda d=d:get_data_dicts(img_dir=img_dir,
                                                             annotation_dir=annotation_dir,
                                                             text_dir=text_dir))
    MetadataCatalog.get(args.image_name).set(things_classes=category_list)

# Prepare Test Set
for d in [""]:
    DatasetCatalog .register(args.image_name + "wide_test/", lambda d=d:get_predict_dicts(test_dir))
    MetadataCatalog.get(args.image_name).set(things_classes=category_list)


# Config
# Config의 값들을 바꿔줌으로써 model을 자유롭게 조정
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = (args.image_name+'wide_train/',)
cfg.DATASETS.TEST = ()
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = args.batch
cfg.SOLVER.BASE_LR = args.lr
cfg.SOLVER.MAX_ITER = args.iter
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 6
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(category_list)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = My_Trainer(cfg)
trainer.resume_or_load(resume=False)

# Training and reload saved model
# trainer.resume_or_load()  # load last checkpoint or MODEL.WEIGHTS
trainer.train()

# # Save Model
os.rename("./output/model_final.pth", "./output/model_lr:%f_expnum:%02i_iter:%i"%(args.lr, args.expnum, args.iter))


# Set MetaData
eye_check_metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])

# 전체 걸린 시간
total_time = time.time()

# 이미지 저장하는데 걸린 시간
image_save_time = time.time()

# Test
# 학습한 model로 test
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_lr:%f_expnum:%02i_iter:%i"%(args.lr, args.expnum, args.iter))
# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_lr:%f_expnum:%02i_iter:%i"%(args.lr, 3, 20000))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.DATASETS.TEST = (args.image_name + "wide_test/",)
predictor = DefaultPredictor(cfg)

i = 1

# print("creating img files")
# if not os.path.exists("./img%02i/"%(args.expnum)):
#     os.mkdir("./img%02i/"%(args.expnum))
#     print("Generate file img%02i"%(args.expnum))

# get category data
# get_output_image(predictor=predictor, image_dir=test_dir, metadata=eye_check_metadata, text_dir=text_dir, expnum=args.expnum)
rename_file(predictor=predictor, image_dir=test_dir, metadata=eye_check_metadata, text_dir=text_dir)
print("rename!!")
# print("Generate Image file!!!")

# print; image save time
image_save_time = time.time() - image_save_time
print("---Total Image save time : %03f sec---" %(image_save_time))

# # text file로 변환 후 저장하는데 걸린 시간
# text_file_time = time.time()

# # save txt file (OCR)
# extract_txt_add_csv(image_dir=test_dir, text_dir=text_dir, expnum=args.expnum)
# print("Generate text file!!!")

# # print; text file time
# text_file_time = time.time() - text_file_time
# print("---Generate text file time : %03f sec---" %(text_file_time))

# print; total time
total_time = time.time() - total_time
print("---Total time : %03f sec---" %(total_time))
# print("---Total time per image : %03f sec/img---" %(total_time/len(test_list)))