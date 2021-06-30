import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from PIL import Image,ImageDraw
import imageio
import glob
import os
from model import Yolov1
from dataset import Dataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
import argparse


class Compose1(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img= t(img)

        return img


transform1 = Compose1([transforms.Resize((448, 448)), transforms.ToTensor()])

# Hyperparameters etc.
seed = 123
torch.manual_seed(seed) 
LEARNING_RATE = 2e-5
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
WEIGHT_DECAY = 0
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = True
LOAD_MODEL_FILE = "overfit_new.pth.tar"
model = Yolov1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
loss_fn = YoloLoss()
if LOAD_MODEL:
    load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, metavar='N',
                        help='please enter the correct path of the video')
    parser.add_argument('--video_name', type=str, metavar='N',
                        help='please enter the name of the video')

    parser.set_defaults(use_ce=1)
    args = parser.parse_args()
    
    return args


#This function will draw the rectangles on the new video frames according to the parameters learned from out trained model

def plot_image1(image, boxes,path,img_nm,video):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
        #print(im.shape)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    
    # Display the image
    #ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    img2=cv2.imread(path)
    h,w,d=img2.shape
    for box in boxes:
        clss=box[0]
        if clss==1:
            print('book')
            box = box[2:]
            assert len(box) == 4, "Got more values than in x, y, h, w, in a box!"
            upper_left_x = box[0] - box[3] / 2
            upper_left_y = box[1] - box[2] / 2
            lower_right_x = box[0] + box[3] / 2
            lower_right_y = box[1] + box[2] / 2
            shape=[(upper_left_x,upper_left_y),(lower_right_x,lower_right_y)]
            
            
            img2=cv2.rectangle(img2,(int(shape[0][0]*w),int(shape[0][1]*h)),(int(shape[1][0]*w),int(shape[1][1]*h)),(0,255,0,255),7)
        if clss==2:
            print('author')
            box = box[2:]
            assert len(box) == 4, "Got more values than in x, y, h, w, in a box!"
            upper_left_x = box[0] - box[3] / 2
            upper_left_y = box[1] - box[2] / 2
            lower_right_x = box[0] + box[3] / 2
            lower_right_y = box[1] + box[2] / 2
            shape=[(upper_left_x,upper_left_y),(lower_right_x,lower_right_y)]
            
            
            img2=cv2.rectangle(img2,(int(shape[0][0]*w),int(shape[0][1]*h)),(int(shape[1][0]*w),int(shape[1][1]*h)),(0,0,255,255),7)
        if clss==3:
            print('title')
            box = box[2:]
            assert len(box) == 4, "Got more values than in x, y, h, w, in a box!"
            upper_left_x = box[0] - box[3] / 2
            upper_left_y = box[1] - box[2] / 2
            lower_right_x = box[0] + box[3] / 2
            lower_right_y = box[1] + box[2] / 2
            shape=[(upper_left_x,upper_left_y),(lower_right_x,lower_right_y)]
            
            

            img2=cv2.rectangle(img2,(int(shape[0][0]*w),int(shape[0][1]*h)),(int(shape[1][0]*w),int(shape[1][1]*h)),(255,0,0,255),7)        
    #os.mkdir("/content/drive/MyDrive/Amphan_book_seller/YOLO/final_submission/v_data_frames/"+video)
    cv2.imwrite("/content/drive/MyDrive/Amphan_book_seller/YOLO/final_submission/v_data_frames/"+video+"/"+img_nm+".jpg",img2)


def make_video(in_folder,out_folder):
  fp_in=in_folder
  fp_out=out_folder
  img_array = []
  for filename in glob.glob(fp_in):
    img = cv2.imread(filename)
    print(img.shape)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
 
  
  out = cv2.VideoWriter(out_folder,cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
 
  for i in range(len(img_array)):
      out.write(img_array[i])
  out.release()


def pred(path,img_nm,video):
  img_nm=img_nm[:4]
  
  image = Image.open(path)
  imge=transform1(image)
  a=imge
  l=[]
  for i in range(BATCH_SIZE):
    l.append(a)
  a=torch.stack(l)

  img=a

  img = img.to(DEVICE)
  bboxes = cellboxes_to_boxes(model(img))
  bboxes = non_max_suppression(bboxes[0], iou_threshold=0.5, threshold=0.4, box_format="midpoint")

  plot_image1(img[0].permute(1,2,0).to("cpu"), bboxes,path,img_nm,video)
  #plot_image1(imge, bboxes)
  #print(bboxes)

#extracting the frames from the videos :

def extractFrames(pathIn, pathOut):
    os.mkdir(pathOut)
    cap = cv2.VideoCapture(pathIn)
    count = 0
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            print('Read %d frame: ' % count, ret)
            cv2.imwrite(os.path.join(pathOut, "{:04d}.jpg".format(count)), frame)  # save frame as JPEG file
            count += 1
        else:
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main(args):
    path1='/content/drive/MyDrive/Amphan_book_seller/YOLO/final_submission/extracted_frames/'+args.video_name
    print(args.video_name)
    extractFrames(args.input_path+args.video_name+'.mp4',path1)
    
    path='/content/drive/MyDrive/Amphan_book_seller/YOLO/final_submission/extracted_frames/'+args.video_name+'/'
    f=os.listdir(path)
    i=cv2.imread(path+f[1])
    if(i.shape[0]<i.shape[1]):
      for file_name in os.listdir(path):
        if file_name.split(".")[-1].lower() in {"jpeg", "jpg", "png"}:
          img=cv2.imread(path+"/"+file_name)
          image = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
          #image = cv2.rotate(img, cv2.cv2.ROTATE_180)
          cv2.imwrite(path1+"/"+file_name,image)
      
    
    #This portion will take each and every frame extracted from the video and then pass it through the prediction function
    imgs=os.listdir(path1)
    for i in range(len(imgs)):
    #print(str(imgs[i]))
      pred(path1+"/"+str(imgs[i]),str(imgs[i]),args.video_name)

    p_img_path="/content/drive/MyDrive/Amphan_book_seller/YOLO/final_submission/v_data_frames/"+args.video_name+"/*"
    make_video(p_img_path,"/content/drive/MyDrive/Amphan_book_seller/YOLO/final_submission/v_data_frames/project{0}.avi".format(args.video_name))

if __name__=="__main__":
    args=parse_args()
    main(args)

    