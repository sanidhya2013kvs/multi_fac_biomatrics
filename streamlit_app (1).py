import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import face_recognition
from datetime import datetime
import os
import dlib
from PIL import Image
import numpy as np
import pandas as pd
img_file = st.sidebar.file_uploader(label='Upload a file', type=['png', 'jpg'],accept_multiple_files=True)
col1, col2 = st.columns([3, 1])
counter=0
if img_file:
  roll_no=[]
  date_time=[]
  
  nameList=[]
  date_time=[]
  #filepath='doc.csv'
  #df1 = pd.read_csv(filepath)
  #st.dataframe(df1)
  def markAttendance(name):
    if name not in nameList:
      nameList.append(name)
      roll_no.append(name)
      now=datetime.now()
      dtsring=now.strftime('%H:%M:%S')
      date_time.append(dtsring)
      #print(nameList)
      #print(date_time)
      
          
          #new_data = {'roll_no': name, 'DATE_Time': dtsring}
          #df1 = pd.read_csv(filepath)
                #df1.append(df)
                #df1.to_csv(filepath)
     

  
  images=[]
  myLIST=[]
  classNames=[]
  for nameo in img_file:
    art=nameo.name
    tempu=os.path.splitext(art)
    Tari=tempu[0]
    myLIST.append(Tari)
    classNames.append(Tari)
    img = Image.open(nameo)
    img = img.save("img.jpg")
    img = cv2.imread("img.jpg")
    #st.image(img)
    images.append(img)


  #st.write(classNames)
  #st.write(myLIST)
  #for cl in img_file:
  
    #image.append(cl)
  #images=[]
  #for ionl in images:
    #st.image(ionl)
    #airte=ionl.name
   # st.write(airte)
    
  #for iot in images:
    
  def findEncodings(images):
    encodeList=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

  encodeListKnown=findEncodings(images)

  #st.write(len(encodeListKnown))
 
  with col1:
    class VideoProcessor:
      def recv(self, frame):
        dict = {'roll_no': roll_no, 'DATE_Time': date_time}
        df = pd.DataFrame(dict) 
        df.to_csv("doc.csv") 
        img = frame.to_ndarray(format="bgr24")
        imgS=cv2.resize(img,(0,0),None,0.25,0.25)
        imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

        faceCurFrame=face_recognition.face_locations(imgS)
        encodeCurFrame=face_recognition.face_encodings(imgS,faceCurFrame)

        for encodeFace,faceLoc in zip (encodeCurFrame,faceCurFrame):
          matches=face_recognition.compare_faces(encodeListKnown,encodeFace)
          facedis=face_recognition.face_distance(encodeListKnown,encodeFace)

          matchindex=np.argmin(facedis)
          #print(facedis)
          if facedis[matchindex]<0.51:
            if matches[matchindex]:
              name=classNames[matchindex].upper()
              #print(name)
              markAttendance(name)
              y1,x2,y2,x1=faceLoc
              y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4
              cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
              y1, x2, y2, x1 = faceLoc
              y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
              cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
          else:
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
      
        return av.VideoFrame.from_ndarray(img, format="bgr24")
      
    ctx=webrtc_streamer(
      key="example",
      video_processor_factory=VideoProcessor,
      rtc_configuration={  
          "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
              })
 
  
  if ctx.state.playing==False:
    import os.path

    file_exists = os.path.exists('doc.csv')
    if file_exists==True:
    #print(nameList)
      with open('doc.csv','r+') as f:
        st.download_button('Download CSV', f,file_name='file.csv')
