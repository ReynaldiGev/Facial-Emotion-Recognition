import time
import requests
import csv
import streamlit as st
from streamlit_lottie import st_lottie
from streamlit_lottie import st_lottie_spinner
from operator import truediv
from re import template
import tempfile
from textwrap import fill
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import seaborn as sns
import glob as gb
import os
from PIL import Image
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image
import cv2
import moviepy.editor as moviepy
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, VideoProcessorBase, WebRtcMode

train_data_dir='dataset/train'
validation_data_dir='dataset/validation'
folder_path = 'D:/Facial-Emotion-Recognition-main/'

temp_file_to_save = './temp_file_1.mp4'
temp_file_result  = './temp_file_2.mp4'
temp_file_emot = './temp_file_3.csv'
temp_file_emot2 = './temp_file_4.txt'


st.set_page_config(page_title="DSS Project", page_icon="📈")

def get_counts(path):
  emotions = os.listdir(path)

  cls_counts = {}
  for emotion in emotions:
    count = len(os.listdir(os.path.join(path, emotion)))
    # print(emotion, count)
    cls_counts[emotion] = count

  return cls_counts
train_counts = get_counts(train_data_dir)

def eda_chart():
    fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, figsize = (14, 8))
    explode = [0.05] * 5
    # Pie for training
    ax0.set_title('Train classes Pie chart')
    ax0.pie(train_counts.values(), labels=train_counts.keys(), explode=explode, autopct='%1.1f%%', shadow=True)
    ax1.set_title('Train classes Bar chart')
    ax1.bar(train_counts.keys(), train_counts.values(), width=0.8)
    st.pyplot(fig)

def sample_angry():
    expression='angry'
    pic_size=48
    fig = plt.figure(figsize=(20,20))
    for i in range(1,19,1):
        plt.subplot(6,6,i)
        img= tf.keras.preprocessing.image.load_img(train_data_dir+'/'+expression+"/"+ os.listdir(train_data_dir+'/'+ expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img)
    st.pyplot(fig)

def sample_happy():
    expression='happy'
    pic_size=48
    fig = plt.figure(figsize=(20,20))
    for i in range(1,19,1):
        plt.subplot(6,6,i)
        img= tf.keras.preprocessing.image.load_img(train_data_dir+'/'+expression+"/"+ os.listdir(train_data_dir+'/'+ expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img)
    st.pyplot(fig)

def sample_neutral():
    expression='neutral'
    pic_size=48
    fig = plt.figure(figsize=(20,20))
    for i in range(1,19,1):
        plt.subplot(6,6,i)
        img= tf.keras.preprocessing.image.load_img(train_data_dir+'/'+expression+"/"+ os.listdir(train_data_dir+'/'+ expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img)
    st.pyplot(fig)

def sample_sad():
    expression='sad'
    pic_size=48
    fig = plt.figure(figsize=(20,20))
    for i in range(1,19,1):
        plt.subplot(6,6,i)
        img= tf.keras.preprocessing.image.load_img(train_data_dir+'/'+expression+"/"+ os.listdir(train_data_dir+'/'+ expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img)
    st.pyplot(fig)

def sample_surprise():
    expression='surprise'
    pic_size=48
    fig = plt.figure(figsize=(20,20))
    for i in range(1,19,1):
        plt.subplot(6,6,i)
        img= tf.keras.preprocessing.image.load_img(train_data_dir+'/'+expression+"/"+ os.listdir(train_data_dir+'/'+ expression)[i],target_size=(pic_size,pic_size))
        plt.imshow(img)
    st.pyplot(fig)



def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def write_bytesio_to_file(filename2, bytesio):
            """
            Write the contents of the given BytesIO to a file.
            Creates the file or overwrites the file if it does
            not exist yet. 
            """
            with open(filename2, "wb") as outfile:
            # Copy the BytesIO stream to the output file
                outfile.write(bytesio.getbuffer())

face_classifier = cv2.CascadeClassifier(r'D:\Facial-Emotion-Recognition-main\haarcascade_frontalface_default.xml')
classifier =load_model(r'D:\Facial-Emotion-Recognition-main\Emotion_model_first.h5')
emotion_labels = ['Angry','Happy','Neutral', 'Sad', 'Surprise']

lottie_url_download = "https://assets1.lottiefiles.com/packages/lf20_KO31Fh.json"
lottie_download = load_lottieurl(lottie_url_download)

model=load_model(r'D:\Facial-Emotion-Recognition-main\Emotion_model_first.h5')
faceDetect=cv2.CascadeClassifier(r'D:\Facial-Emotion-Recognition-main\haarcascade_frontalface_default.xml')
labels_dict= {0:'Angry',1:'Happy',2:'Neutral',3:'Sad',4:'Surprise'}

def image_face_detected(image_in):
    frame=cv2.imread(image_in)
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces= faceDetect.detectMultiScale(gray, 1.3, 3)
    get_label = []
    tempathasil = './fotohasil.jpg'
    for x,y,w,h in faces:
        sub_face_img=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_img,(48,48))
        normalize=resized/255.0
        reshaped=np.reshape(normalize, (1, 48, 48, 1))
        
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        emot = labels_dict[label]
        
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
        cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        if emot !=0:
            get_label.append(emot)
        else:
            print('abc')
        cv2.imwrite(tempathasil, frame)

    buka = Image.open(tempathasil)
    #cv2.imshow("Frame",frame)
    st.image(buka)
    df = pd.DataFrame(get_label)
    df.columns = ["Expression"]
    df = df.value_counts().rename_axis('Expression').reset_index(name='counts')

    labels = pd.DataFrame(emotion_labels)
    labels.columns = ['Expression']

    df2 = pd.merge(labels, df, on='Expression', how='left')
    df2['counts'] = df2['counts'].fillna(0)

    df2.to_csv(temp_file_emot, index=False)



RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

class Faceemotion(VideoTransformerBase):
                    def transform(self, frame):
                        img = frame.to_ndarray(format="bgr24")

                        img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                        faces = face_classifier.detectMultiScale(img_gray)
                        f=open(temp_file_emot2, 'a')

                        for (x,y,w,h) in faces:
                            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,255),2) 
                            roi_gray = img_gray[y:y+h,x:x+w]
                            roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
      
                            if np.sum([roi_gray])!=0:
                                roi = roi_gray.astype('float')/255.0  # normalizing
                                roi = tf.keras.preprocessing.image.img_to_array(roi)
                                roi = np.expand_dims(roi,axis=0)

                                prediction = classifier.predict(roi)[0]
                                label=emotion_labels[prediction.argmax()]
                                f.write(label+"\n")
                                label_position = (x,y-10)
                                cv2.putText(img,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                            else:
                                cv2.putText(img,'No Faces Found',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                    

                            
                        return img


def statistics_visualization():
    df2 = pd.read_csv(temp_file_emot)
    fig = go.Figure(data=go.Scatterpolar(r=df2['counts'],
      theta=df2['Expression'],
      fill='toself',
      hovertemplate = "<br>Emotion: %{theta} </br> Count: %{r} </br> "
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True
        ),
      ),
      showlegend=False
    )
    
    st.write("Emotion Radar",fig)

    fig2 = go.Figure(go.Bar(x=df2['counts'], 
                       y=df2['Expression'], 
                       orientation='h',
                       hovertemplate = "<br>Jumlah: %{x} </br>" ))
    fig2.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
    st.write("Emotion Barplot",fig2)

def statistics_visualizationtwo():
    txt =  pd.read_fwf('./temp_file_4.txt', header=None, names=['Expression'])
    df=pd.DataFrame(txt)
    df = df.value_counts().rename_axis('Expression').reset_index(name='counts')

    emotion_labels = ['Angry','Happy','Neutral', 'Sad', 'Surprise']
    labels = pd.DataFrame(emotion_labels)
    labels.columns = ['Expression']
    df2 = pd.merge(labels,df, on = 'Expression', how='left')
    df2['counts'] = df2['counts'].fillna(0)

    fig = go.Figure(data=go.Scatterpolar(r=df2['counts'],
      theta=df2['Expression'],
      fill='toself',
      hovertemplate = "<br>Emotion: %{theta} </br> Count: %{r} </br> "
    ))

    fig.update_layout(
      polar=dict(
        radialaxis=dict(
          visible=True
        ),
      ),
      showlegend=False
    )
    
    st.write("Emotion Radar",fig)

    fig2 = go.Figure(go.Bar(x=df2['counts'], 
                       y=df2['Expression'], 
                       orientation='h',
                       hovertemplate = "<br>Jumlah: %{x} </br>" ))
    fig2.update_layout(barmode='stack', yaxis={'categoryorder':'total ascending'})
    st.write("Emotion Barplot",fig2)




def main():
    st.title("Face Emotion Detection Application")
    st.sidebar.header("Menu")
    activiteis = ["Introduction", "Dataset", "Run App","About"]
    choice = st.sidebar.selectbox("Select Activity", activiteis)

    if choice == "Introduction":
        st.write("""This demo illustrates a combination of plotting and animation with
        Streamlit. We're generating a bunch of random numbers in a loop for around 5 seconds. Enjoy!""")
        html_temp_home1 = """<div style="background-color:#6D7B8D;padding:10px">
                                            <h4 style="color:white;text-align:center;">
                                            Face Emotion detection application using OpenCV, Custom CNN model and Streamlit.</h4>
                                            </div>
                                            </br>"""
        st.markdown(html_temp_home1, unsafe_allow_html=True)
        st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)
        tab1, tab2, tab3 = st.tabs(["📈 FER", "🗃 CNN","Why Important?"])
        with tab1:
            st.header("Face Emotion Recognition")
            st.image("https://recfaces.com/wp-content/uploads/2021/03/rf-emotion-recognition-rf-830x495-1.jpeg", width=600)
            st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)
        with tab2:
            st.header("Convolutional Neural Network")
            st.image("https://www.jeremyjordan.me/content/images/2018/04/vgg16.png", width=600)
            st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)
        with tab3:
            st.write("""
                 Bisnis impact.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)

    elif choice == "Dataset":
        st.header("1. Data darimana?")
        st.write("""
                 The application has two functionalities.
                 1. Real time face detection using web cam feed.
                 2. Real time face emotion recognization.
                 """)
        st.header("2. Amount of Data")

        col1, col2 = st.columns(2)
        with col1:
            traind = st.checkbox('Train Data')
            if traind:
                for folder in os.listdir(train_data_dir):
                    files = gb.glob(pathname= str(train_data_dir+ '/'+ folder + '/*.jpg'))
                    st.write(f'Found {len(files)} images in folder {folder}')
            
        with col2:
            testd = st.checkbox('Test Data')
            if testd:
                for folder in os.listdir(validation_data_dir):
                    files = gb.glob(pathname= str(validation_data_dir+ '/'+ folder + '/*.jpg'))
                    st.write(f'Found {len(files)} images in folder {folder}')

        eda_chart()

        st.header("3. Image Specifications")
        for folder in os.listdir(train_data_dir):
            files = gb.glob(pathname= str(train_data_dir+ '/'+ folder + '/*.jpg'))
        im = Image.open(files[1])
        st.write(im.format, im.size, im.mode)
        with st.expander("See explanation"):
            st.write("""
                 Image Spesification based on PILLOW documentation https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes.
                 1. Images format in JPEG.
                 2. Images dimensions 48x48 pixels.
                 3. L means 8-bit pixels with 2 color black and white (gray scale) on range 0-255.
                 """)

        st.header("4. Display Sample Gambar")
        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Angry", "Happy","Neutral","Sad","Surprise"])
        with tab1:
            sample_angry()
        with tab2:
            sample_happy()
        with tab3:
            sample_neutral()
        with tab4:
            sample_sad()
        with tab5:
            sample_surprise()
        
    elif choice == "Run App":
        st.sidebar.subheader("Pick your channel ")
        mediapg = st.sidebar.radio("How would you like to use?",["Video","Image","Live Camera"])
        if mediapg == "Video":
             uploaded_video = st.file_uploader("Upload Video", type = ['mp4','mpeg','mov','webm'])
             def video():
                write_bytesio_to_file(temp_file_to_save, uploaded_video)
                cap = cv2.VideoCapture(temp_file_to_save)
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_fps = cap.get(cv2.CAP_PROP_FPS)  
                st.write(width, height, frame_fps)
                out = cv2.VideoWriter(temp_file_result, fourcc, frame_fps, (width, height))
                get_label = []
                while True:
                    ret, frame=cap.read()
                    if not ret: break
                    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
                    faces=face_classifier.detectMultiScale(gray, 1.3, 5)
                    
                    for (x,y,w,h) in faces:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2) 
                        roi_gray = gray[y:y+h,x:x+w]
                        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)
                
                        if np.sum([roi_gray])!=0:
                            roi = roi_gray.astype('float')/255.0  
                            roi = tf.keras.preprocessing.image.img_to_array(roi)
                            roi = np.expand_dims(roi,axis=0)

                            prediction = classifier.predict(roi)[0]
                            label=emotion_labels[prediction.argmax()]
                            label_position = (x,y-10)
                            cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        else:
                            cv2.putText(frame,'No Faces Found',(30,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                        
                        if label !=0:
                            get_label.append(label)
                        else:
                            print('abc')
                    #cv2.imshow('Emotion Detector',frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                    out.write(frame)
                    
                out.release() 
                cap.release()
                cv2.destroyAllWindows()

                output_video = open(temp_file_result,'rb')
                out_bytes = output_video.read()
                st.video(out_bytes)
                st.write("Detected Video")

                df = pd.DataFrame(get_label)
                df.columns = ["Expression"]
                df = df.value_counts().rename_axis('Expression').reset_index(name='counts')
                labels = pd.DataFrame(emotion_labels)
                labels.columns = ['Expression']

                df2 = pd.merge(labels, df, on='Expression', how='left')
                df2['counts'] = df2['counts'].fillna(0)
                df2.to_csv(temp_file_emot, index=False)

             if uploaded_video != None:
                vid = uploaded_video.name
                with open(vid, mode='wb') as f:
                    f.write(uploaded_video.read()) # save video to disk
                
                st_video = open(vid,'rb')
                video_bytes = st_video.read()
                st.video(video_bytes)
                st.write("Uploaded Video")
                st.text("Press Process to display the face emotion detected image.")
                if st.button('Process', key='pross'):
                    with st_lottie_spinner(lottie_download):
                        video()
                    with st.expander("See Result"):
                        statistics_visualization()


        elif mediapg == "Image":
            uploaded_image = st.file_uploader("Upload Image", type = ['jpg','png','jpeg'])
            if uploaded_image is not None:
                image1 = Image.open(uploaded_image)
                st.image(image1)
                st.text('Uploaded Image')
                #st.write(uploaded_image.name)

                st.text('Press Process to display the face emotion detected image')
                if st.button('Process'):
                    image_face_detected(uploaded_image.name)
                    with st.expander("See Result"):
                        statistics_visualization()

        elif mediapg =="Live Camera":
            with open(temp_file_emot2, 'a'):pass
            st.header("Live Stream")
            st.write("Click on start to use webcam and detect your face emotion")
            webrtc_streamer(key="example", video_processor_factory=Faceemotion,mode=WebRtcMode.SENDRECV, rtc_configuration=RTC_CONFIGURATION)
            if st.button('Process'):
                with st.expander("See Result"):
                    statistics_visualizationtwo()


# df = pd.DataFrame(get_label)
# df.columns = ["Expression"]
# df = df.value_counts().rename_axis('Expression').reset_index(name='counts')
# labels = pd.DataFrame(emotion_labels)
# labels.columns = ['Expression']

# df2 = pd.merge(labels, df, on='Expression', how='left')
# df2['counts'] = df2['counts'].fillna(0)
# df2.to_csv(temp_file_emot, index=False)

if __name__ == "__main__":
    main()
