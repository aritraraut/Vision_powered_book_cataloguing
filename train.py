"""
Main file for training Yolo model on our book dataset dataset

"""

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import Dataset
from utils import (
    non_max_suppression,
    mean_average_precision,
    intersection_over_union,
    cellboxes_to_boxes,
    get_bboxes,
    plot_image,
    save_checkpoint,
    load_checkpoint,
)
from loss import YoloLoss
import matplotlib.pyplot as plt

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 64 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 0
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_BEST_MAP = False
LOAD_MODEL_FILE = "overfit_new.pth.tar"
IMG_DIR = "data/images"
LABEL_DIR = "data/labels"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes


transform = Compose([transforms.Resize((448, 448)), transforms.ToTensor(),])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean training loss was {sum(mean_loss)/len(mean_loss)}")
    return str(sum(mean_loss)/len(mean_loss))


def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = Dataset(
        "train3.csv",
        transform=transform,
        img_dir=IMG_DIR,
        label_dir=LABEL_DIR,
    )

    test_dataset = Dataset(
        "test3.csv", transform=transform, img_dir=IMG_DIR, label_dir=LABEL_DIR,
    )

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )
    
    if LOAD_BEST_MAP==True:
      print("hello")
      with open('myfile.txt', 'r') as file:
        input_lines = [line.strip() for line in file]
      best_map=float(input_lines[0][7:13])
    else:  
      best_map=.30
    
    for epoch in range(EPOCHS):
        # for x, y in train_loader:
        #    x = x.to(DEVICE)
        #    for idx in range(8):
        #        bboxes = cellboxes_to_boxes(model(x))
        #        bboxes = non_max_suppression(bboxes[idx], iou_threshold=0.5, threshold=0.4, box_format="midpoint")
        #        plot_image(x[idx].permute(1,2,0).to("cpu"), bboxes)

        #    import sys
        #    sys.exit()
        print("------------------------------------")
        print("epoch:"+str(epoch))

        #training ::::
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.5, threshold=0.4
        )

        mean_avg_prec = mean_average_precision(
            pred_boxes, target_boxes, iou_threshold=0.5, box_format="midpoint"
        )
        print(f"Train mAP: {mean_avg_prec}")
        # mean_avg_prec=str(mean_avg_prec[7:])
        # mean_avg_prec=str(mean_avg_prec[:-2])
        yo=str(mean_avg_prec)[7:]
        yo=yo[:-1]
      
        f_map=open("mean_avg_precision.txt","a")
        f_map.writelines([yo," "])
        f_map.close()

        if mean_avg_prec > best_map:
           best_map=mean_avg_prec
           file1 = open("myfile.txt","w")
           file1.writelines([str(best_map)])
           file1.close()
           checkpoint = {
               "state_dict": model.state_dict(),
               "optimizer": optimizer.state_dict(),
           }
           save_checkpoint(checkpoint, filename=LOAD_MODEL_FILE)
           import time
           time.sleep(10)

        t_loss=train_fn(train_loader, model, optimizer, loss_fn)
        #training_loss.append(t_loss)
        f_tl=open("training_loss.txt","a")
        f_tl.writelines([t_loss," "])
        f_tl.close()

        #validation::::
        loop1 = tqdm(test_loader, leave=True)
        mean_valid_loss = []
        for batch_idx, (x, y) in enumerate(loop1):
          x, y = x.to(DEVICE), y.to(DEVICE)
          out = model(x)
          loss = loss_fn(out, y)
          mean_valid_loss.append(loss.item())
          # update progress bar
          loop1.set_postfix(loss=loss.item())

        print(f"Mean validation loss was {sum(mean_valid_loss)/len(mean_valid_loss)}")
        validation_loss=(str(sum(mean_valid_loss)/len(mean_valid_loss)))
        f_vl=open("validation_loss.txt","a")
        f_vl.writelines([validation_loss," "])
        f_vl.close()



if __name__ == "__main__":
    main()
