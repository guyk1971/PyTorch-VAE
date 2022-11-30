# this file includes utilities that are used by deep_fashion.ipynb notebook



import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import pandas as pd
from glob import glob as Glob
import cv2 
import json



def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


def show_img_bbox(imgname,bbox):
    img=Image.open(imgname)
    img = np.array(img)
    # bb = bbox.loc[imgname.split('Img/')[1]]
    bb = bbox       # assuming we get the bb for the imgname 
    img = cv2.rectangle(img,tuple(bb[:2]),tuple(bb[2:]),(255,0,0),2)
    return Image.fromarray(img)


def crop_img_bbox(imgname,bbox=None,size=256,value=(255,255,255)):
    img=Image.open(imgname)
    # x1,y1,x2,y2 = bbox.loc[imgname.split('Img/')[1]] # x1 y1 x2 y2
    if bbox is not None:
        x1,y1,x2,y2 = bbox
    else:
        x1,y1=0,0
        x2,y2=img.size 
    img = np.array(img)
    w=x2-x1
    h=y2-y1
    rsz = size/h if h>w else size/w
    w=int(w*rsz)
    h=int(h*rsz)
    new_img = cv2.resize(img[y1:y2,x1:x2],(w,h))
    new_img = cv2.copyMakeBorder(new_img,(size-h)//2,size-h-(size-h)//2,(size-w)//2,size-w-(size-w)//2,cv2.BORDER_CONSTANT,value = (255,255,255))
    new_img=Image.fromarray(new_img)    
    return new_img

def show_avg_img(imgs_paths):
    img_indxs = [i for i in range(len(imgs_paths))]
    avg_img = np.zeros_like(Image.open(imgs_paths[0]),dtype=np.float64)
    for i,idx in enumerate(img_indxs):
        img=Image.open(imgs_paths[idx])
        avg_img+=np.asarray(img)
    avg_img = avg_img/len(img_indxs)
    return Image.fromarray(np.uint8(avg_img))



# Category and Attribute Prediction (CATP)
def catp_read_bbox_to_df(bbox_path):
    with open(bbox_path,'r') as f:
        bbox=f.read().splitlines()
    cols=bbox[1].split()
    bbox=bbox[2:]
    bbox=[i.split() for i in bbox]
    bbox = {i[0]:[int(a) for a in i[1:]] for i in bbox}
    bbox = pd.DataFrame.from_dict(bbox,orient='index',columns=cols[1:])
    return bbox

# In shop Clothes Retrival (ISCR)



if __name__=='__main__':
    if os.uname()[1]=='guy-x':
        dataset_path = '/home/guy/sd1tb/datasets/deep_fashion/DeepFashion/Category and Attribute Prediction Benchmark'
    #     dataset_path = '/home/guy/sd1tb/datasets/deep_fashion/DeepFashion/In-shop Clothes Retrieval Benchmark'
    else:   # assuming gpu15
        dataset_path = '/data/users/gkoren2/datasets/DeepFashion/Category-and-Attribute-Prediction' # gpu15
    #     dataset_path = '/data/users/gkoren2/datasets/DeepFashion/In-shop-Clothes-Retrieval/In-shop Clothes Retrieval Benchmark' # gpu15

    print(f"assuming we're on {os.uname()[1]} so data is in {dataset_path}")

    img_fn=[i for i in Glob(f'{dataset_path}/Img/img/**/*.jpg',recursive=True)]      # Category-and-Attribute-Prediction

    bbox_path = os.path.join(dataset_path,'Anno_coarse','list_bbox.txt')
    bbox_df=catp_read_bbox_to_df(bbox_path)

    idx=0
    img=img_fn[idx]
    bbox = bbox_df.loc[img.split('Img/')[1]]

    dst_img = crop_img_bbox(img,bbox)

    print(dst_img.size)

