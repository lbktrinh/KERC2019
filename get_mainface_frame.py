import argparse, cv2, numpy as np, sys, os
from moviepy.editor import *
import csv
import os

from prlab.detection.tinyface.detection import TinyFaceDetection
from prlab.utils.video import VideoReader



            
def key_faces(bboxes):
    key_faces = []
    if len(bboxes)>0:
        key_cmp   = []
        key_position_x =[]
        key_position_y =[]
        key_width =[]
        key_height =[]
        
        key_id    = list(range(len(bboxes)))
        for i, d in enumerate(bboxes):
            (x, y, w, h) = (int(d[0]), int(d[1]), int(d[2]), int(d[3]))
            key_cmp.append(w*h)
            key_position_x.append(x)
            key_position_y.append(y)
            key_width.append(w)
            key_height.append(h)
        key_faces = list(zip(key_cmp, key_position_x, key_position_y, key_width, key_height))
        key_faces.sort(reverse=True)
    return key_faces[0]
  


              
path     = 'C:/Users/trinhle/Desktop/CODE_challenge/data_example/data_try/'
path_out = 'C:/Users/trinhle/Desktop/CODE_challenge/data_example/data_try_out/'

data_split = os.listdir(path)


for data_portion in data_split:
    all_label = os.listdir(os.path.join(path,data_portion))

    for label in all_label:

        allfiles = os.listdir(os.path.join(path,data_portion,label))
         

        for fname in allfiles:
            if fname.endswith('.mp4'):

                print(fname)

                if os.path.exists(os.path.join(path_out,data_portion,label,fname[:-4])) == False:
                    os.makedirs(os.path.join(path_out,data_portion,label,fname[:-4]))
                    
             
                detector = TinyFaceDetection.getDetector()
                
                cap = cv2.VideoCapture(os.path.join(path,data_portion,label,fname))
                i=0
                dem = 0
                while(cap.isOpened()):
                    ret, img = cap.read()
                    if ret == False:
                        break
                      
                    bboxes, _  = detector.detect(img)
              
                    if len(bboxes)>0:
                    
                        dem = dem + 1
                    
                 
                        d = key_faces(bboxes)
                    
                        key_face = [max(0,int(d[1])), max(0,int(d[2])), max(0,int(d[3])), max(0,int(d[4]))]
                        
    
                        (x,y,w,h)= (int(key_face[0]), int(key_face[1]), int(key_face[2]), int(key_face[3]))
                        if img[y:y+h,x:x+w].shape[0]>0 and img[y:y+h,x:x+w].shape[1]>0:
                            k_face=img[y:y+h,x:x+w]
                            cv2.imwrite(os.path.join(path_out,data_portion,label,fname[:-4],'1_{:04d}.jpg'.format(dem)), k_face)
                            print (dem)
                    i+=1
 
                cap.release()
       
