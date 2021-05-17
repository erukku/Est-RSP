import numpy as np

import time
import os
import copy
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import cv2
from matplotlib import pyplot as plt
from pathlib import Path

import cv2

cap = cv2.VideoCapture(0)

data_transform = transforms.Compose([
    transforms.ToTensor()
])

gu = cv2.imread("gu.png")
ch = cv2.imread("ch.png")
pa = cv2.imread("pa.png")

print(gu.shape)
print(ch.shape)
print(pa.shape)
class MyModel(torch.nn.Module):
  def __init__(self,classnum):
    super().__init__()
    self.conv1 = torch.nn.Conv2d(3, 16, (3, 3), stride = 2, padding = 1)
    self.relu = torch.nn.ReLU()
    self.conv2 = torch.nn.Conv2d(16, 32, (3, 3), stride = 2, padding = 1)
    self.conv3 = torch.nn.Conv2d(32, 64, (3, 3), stride = 2, padding = 1)
    self.conv4 = torch.nn.Conv2d(64, 64, (3, 3), stride = 2, padding = 1)
    self.conv5 = torch.nn.Conv2d(64, 64, (3, 3), stride = 2, padding = 1)
    self.flatten = torch.nn.Flatten()
    self.lstm = torch.nn.LSTM(6400, 128, batch_first=True)
    self.fc1 = torch.nn.Linear(128, 256)
    self.fc2 = torch.nn.Linear(256, 64)
    self.fc3 = torch.nn.Linear(64, classnum)
    #self.softmax = torch.nn.Softmax(dim=1)
    
  def forward(self, x):
    batch_size, seq_size, C, W, H = x.size()
    x = x.view(batch_size * seq_size, C, W, H).contiguous()
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.conv3(x)
    x = self.relu(x)
    x = self.conv4(x)
    x = self.relu(x)
    x = self.conv5(x)
    x = self.relu(x)
    x = self.flatten(x)
    x = x.view(batch_size, seq_size, -1).contiguous()
    #x, (h_n, c_n) = self.lstm(x)
    x,_ = self.lstm(x)
    #print(x.size())

    #x = torch.squeeze(x[:, seq_size-1, :])
    x = self.fc1(x)
    #print(x.size())
    x = self.relu(x)
    x = self.fc2(x)
    #print(x.size())
    x = self.relu(x)
    x = self.fc3(x)
    #print(x.size())
    #x = self.softmax(x)
    x = x[:,-1,:]
    #print(x.size())
    #print(x)

    return x
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

loaded_model = torch.load('bbest_model.h5', map_location=torch.device('cpu')) 

f = 0
side = 0
ff = 0
while(1):
  
  _,fr = cap.read()
  fr = cv2.flip(fr,1)
  overlay = fr.copy()
  
  if not f:
    count = 0
    buffer = []
    if side == 1:
      cv2.rectangle(overlay, (0,0), (720,720),(0,0,0), -1)
    else:
      cv2.rectangle(overlay, (560,0), (1280,720),(0,0,0), -1)
    fr = cv2.addWeighted(overlay, 0.4, fr, 1 - 0.4, 0)
    cv2.putText(fr,"left hand",(100, 350),cv2.FONT_HERSHEY_PLAIN, 4,  (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(fr,"right hand",(850, 350),cv2.FONT_HERSHEY_PLAIN, 4,  (255, 255, 255), 3, cv2.LINE_AA)
    
    
    cv2.imshow("gg",fr)
    po = cv2.waitKey(1)
    #print(po)
    if po == 2 or po == 3:
      side ^= 1
    if po == 13:
      f ^= 1
  else:
    if not ff:
      if side == 1:
        cv2.rectangle(overlay, (0,0), (720,720),(0,0,0), -1)
      else:
        cv2.rectangle(overlay, (560,0), (1280,720),(0,0,0), -1)
      fr = cv2.addWeighted(overlay, 0.7, fr, 1 - 0.7, 0)
      if side == 1:
        cv2.putText(fr,"PRESS Return",(30, 300),cv2.FONT_HERSHEY_PLAIN, 6,  (255, 255, 255), 5, cv2.LINE_AA)
      else:
        cv2.putText(fr,"PRESS Return",(600, 300),cv2.FONT_HERSHEY_PLAIN, 6,  (255, 255, 255), 5, cv2.LINE_AA)
      cv2.imshow("gg",fr)
      po = cv2.waitKey(1)
      
      if po == 13:
        ff ^= 1
      
    else:
      mya = 3 - count//30
      if side == 1:
        frame = fr[:720,-561:,:]
        cv2.rectangle(overlay, (0,0), (720,720),(0,0,0), -1)

      else:
        frame = fr[:720,:560,:]
        cv2.rectangle(overlay, (560,0), (1280,720),(0,0,0), -1)

      fr = cv2.addWeighted(overlay, 0.7, fr, 1 - 0.7, 0)
      if count <= 30*3 + 1 + 1:
        if side == 1:
          cv2.putText(fr,str(mya),(280, 300),cv2.FONT_HERSHEY_PLAIN, 15,  (255, 255, 255), 5, cv2.LINE_AA)
        else:
          cv2.putText(fr,str(mya),(850, 300),cv2.FONT_HERSHEY_PLAIN, 15,  (255, 255, 255), 5, cv2.LINE_AA)
      elif count == 92 + 1:
        if side == 1:
          if preds == 0:
            fr[280:280+pa.shape[0],250:250+pa.shape[1]] = pa  
          elif preds == 1:
            fr[280:280+gu.shape[0],250:250+gu.shape[1]] = gu
          else:
            fr[280:280+ch.shape[0],250:250+ch.shape[1]] = ch
          cv2.putText(fr,"Change hand:[c],Continue:Ret,Quit:[q]",(50, 600),cv2.FONT_HERSHEY_PLAIN, 2,  (255, 255, 255), 2, cv2.LINE_AA)
        else:
          if preds == 0:
            fr[280:280+pa.shape[0],800:800+pa.shape[1]] = pa  
          elif preds == 1:
            fr[280:280+gu.shape[0],800:800+gu.shape[1]] = gu
          else:
            fr[280:280+ch.shape[0],800:800+ch.shape[1]] = ch
          cv2.putText(fr,"Change hand:[c],Continue:Ret,Quit:[q]",(620, 600),cv2.FONT_HERSHEY_PLAIN, 2,  (255, 255, 255), 2, cv2.LINE_AA)
      cv2.imshow('gg',fr)
      #cv2.imshow('g',frame)
      po = cv2.waitKey(1)
      if po == 13 and count == 92+1:
        ff = 0
        count = 0
        buffer = []

        continue
      elif po == ord("q") and count == 92+1:
        cap.release()
        cv2.destroyAllWindows()
        exit()
      elif po == ord("c") and count == 92+1:
        f = 0
        ff = 0
        continue
      frame = cv2.medianBlur(frame,5)
      kernel = np.ones((20,20),np.float32)/400
      YCb = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)

      YCb = cv2.medianBlur(YCb,5)
      kernel = np.ones((20,20),np.float32)/400


      #cv2.imshow('frame',YCb)
      YCrCb_mask = cv2.inRange(YCb, (100, 140, 110), (200,160,135)) 
      kernel = np.ones((30,30),np.float32)
      YCrCb_mask = cv2.dilate(YCrCb_mask,kernel,iterations = 1)
      #cv2.imshow("YCb.jpg",YCrCb_mask)
      contours = cv2.findContours(YCrCb_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
      if len(contours) != 0:
        max_cnt = max(contours, key=lambda x: cv2.contourArea(x))
        x,y,w,h = cv2.boundingRect(max_cnt)

        out = np.zeros_like(YCrCb_mask)
        out = cv2.rectangle(out,(x,y),(x+w,720),(255,255,255),-100)
        aaa = frame[y:720,x:x+w,:]
        aaa = cv2.resize(aaa,(300,300))
        masked_img = aaa
        #cv2.imshow("YCbC.jpg",out)
        #cv2.drawContours(out, [max_cnt], -1, color=255, thickness=3)
        #masked_img = cv2.bitwise_and(frame,frame,mask = out)
      else:
        masked_img = cv2.bitwise_and(frame,frame,mask = YCrCb_mask)
        masked_img = cv2.resize(masked_img,(300,300))
      #cv2.imshow("YCbCr.jpg",masked_img)
      if count >= 60+1 and count < 30*3+1:
        masked_img = data_transform(masked_img)
        buffer.append(masked_img)
      elif count == 91+1:
        buffer = torch.stack(buffer).float()
        print(buffer.size())
        buffer = buffer.reshape(1,30, 3, 300, 300)
        output = loaded_model(buffer)
        _, preds = torch.max(output, 1)
        preds = preds[0].item()
        #print(output,preds[0])
        #exit()
        if side == 1:
          cv2.putText(fr,str(mya),(280, 300),cv2.FONT_HERSHEY_PLAIN, 15,  (255, 255, 255), 5, cv2.LINE_AA)
        else:
          cv2.putText(fr,str(mya),(850, 300),cv2.FONT_HERSHEY_PLAIN, 15,  (255, 255, 255), 5, cv2.LINE_AA)
      


      if count != 92+1:
        count += 1
      

          


        
            
                



