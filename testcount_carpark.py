import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model=YOLO('yolov8s.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap=cv2.VideoCapture('park.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")
   

#area1=[(52,364),(30,417),(73,412),(88,369)]
area1=[(29,397),(3,478),(87,483),(122,394)]

area2=[(146,388),(110,476),(214,474),(242,382)]

area3=[(260,378),(239,471),(341,464),(348,375)]

area4=[(370,375),(365,444),(460,454),(467,370)]

area5=[(489,372),(493,448),(591,449),(568,365)]

area6=[(591,364),(617,439),(717,441),(683,363)] 

#area7=[(396,338),(426,404),(479,399),(439,334)]

#area8=[(458,333),(494,397),(543,390),(495,330)]

#area9=[(511,327),(557,388),(603,383),(549,324)]

#area10=[(564,323),(615,381),(654,372),(596,315)]

#area11=[(616,316),(666,369),(703,363),(642,312)]

#area12=[(674,311),(730,360),(764,355),(707,308)]




while True:    
    ret,frame = cap.read()
    if not ret:
        break
    time.sleep(1)
    frame=cv2.resize(frame,(1020,500))

    results=model.predict(frame)
 #   print(results)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
#    print(px)
    list1=[]
    list2=[]
    list3=[]
    list4=[]
    list5=[]
    list6=[]
    #list7=[]
    #list8=[]
    #list9=[]
    #list10=[]
    #list11=[]
    #list12=[]
    
    for index,row in px.iterrows():
#        print(row)
 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'car' in c:
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2

            results1=cv2.pointPolygonTest(np.array(area1,np.int32),((cx,cy)),False)
            if results1>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list1.append(c)
               cv2.putText(frame,str(c),(x1,y1),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
            
            results2=cv2.pointPolygonTest(np.array(area2,np.int32),((cx,cy)),False)
            if results2>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list2.append(c)
            
            results3=cv2.pointPolygonTest(np.array(area3,np.int32),((cx,cy)),False)
            if results3>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list3.append(c)   
            results4=cv2.pointPolygonTest(np.array(area4,np.int32),((cx,cy)),False)
            if results4>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list4.append(c)  
            results5=cv2.pointPolygonTest(np.array(area5,np.int32),((cx,cy)),False)
            if results5>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list5.append(c)  
            results6=cv2.pointPolygonTest(np.array(area6,np.int32),((cx,cy)),False)
            if results6>=0:
               cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
               cv2.circle(frame,(cx,cy),3,(0,0,255),-1)
               list6.append(c)  
            
              
            
    a1=(len(list1))
    a2=(len(list2))       
    a3=(len(list3))    
    a4=(len(list4))
    a5=(len(list5))
    a6=(len(list6)) 
    
    o=(a1+a2+a3+a4+a5+a6)
    space=(6-o)
    print(space)
    if a1==1:
        cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)
        #cv2.putText(frame,str('1'),(50,441),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,255,0),2)
        #cv2.putText(frame,str('1'),(50,441),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a2==1:
        cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,0,255),2)
        #cv2.putText(frame,str('2'),(106,440),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area2,np.int32)],True,(0,255,0),2)
        #cv2.putText(frame,str('2'),(106,440),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a3==1:
        cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,0,255),2)
        #cv2.putText(frame,str('3'),(175,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area3,np.int32)],True,(0,255,0),2)
        #cv2.putText(frame,str('3'),(175,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a4==1:
        cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,0,255),2)
        #cv2.putText(frame,str('4'),(250,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area4,np.int32)],True,(0,255,0),2)
        #cv2.putText(frame,str('4'),(250,436),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a5==1:
        cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,0,255),2)
        #cv2.putText(frame,str('5'),(315,429),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area5,np.int32)],True,(0,255,0),2)
        #cv2.putText(frame,str('5'),(315,429),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
    if a6==1:
        cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,0,255),2)
        #cv2.putText(frame,str('6'),(386,421),cv2.FONT_HERSHEY_COMPLEX,0.5,(0,0,255),1)
    else:
        cv2.polylines(frame,[np.array(area6,np.int32)],True,(0,255,0),2)
        #cv2.putText(frame,str('6'),(386,421),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1) 
    

   
    
    cv2.putText(frame,str(space),(23,30),cv2.FONT_HERSHEY_PLAIN,3,(255,255,255),2)

    cv2.imshow("RGB", frame)

    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
#stream.stop()
