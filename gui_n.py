import tkinter as tk
from tkinter import filedialog
from tkinter import *

from sklearn import metrics
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk
import numpy as np
import cv2

# donwload haarcascade_frontalface_default from here "https://github.com/opencv/opencv/tree/master/data/haarcascades"

def FacialExpressionModel(json_file, weights_file):
    with open(json_file,"r") as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)

    model.load_weights(weights_file)
    model.compile(optimizer ='adam', loss='categorical_crossentropy', metrics = ['accuracy'])

    return model

top =tk.Tk()
top.geometry('800x600')
top.title('Nationality Detector')
top.configure(background='#CDCDCD')

label1 = Label(top, background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)

face = cv2.CascadeClassifier(r"C:/Users/harsh/Desktop/Nationality_Detection/haarcascade_frontalface_default.xml")
model = FacialExpressionModel("C:/Users/harsh/Desktop/Nationality_Detection/model_n.json","C:/Users/harsh/Desktop/Nationality_Detection/model.weights.h5")

EMOTIONS_LIST = ["bhartiya","russian","italian","chinese"]

def Detect(file_path):
    global Label_packed

    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    gray_image = cv2.equalizeHist(gray_image)


    print("File path:", file_path)
    print("Image shape:", image.shape)

    print("Gray image shape:", gray_image.shape)

    faces = face.detectMultiScale(gray_image,scaleFactor=1.1, minNeighbors=6)

    print("Number of faces detected:", len(faces))
    for (x, y, w, h) in faces:
        print("Face coordinates (x, y, w, h):", x, y, w, h)
        


    try:
        for (x,y,w,h) in faces:
            fc = gray_image[y:y+h,x:x+w]
            roi = cv2.resize(fc,(96,96))
            # Convert grayscale to RGB
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_GRAY2RGB)

            # Normalize pixel values
            roi_rgb = roi_rgb / 255.0
            roi_input = np.expand_dims(roi_rgb, axis=0)
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi_input))]
            print("Nationality is " + pred)
            label1.configure(foreground="#011638",text = pred)
    except Exception as e:
        print("Error during detection:", str(e))
        label1.configure(foreground="#011638",text = "Unable to detect")

def show_Detect_button(file_path):
    detect_b = Button(top,text="Detect Nationality", command= lambda: Detect(file_path),padx=10,pady=5)
    detect_b.configure(background="#364156",foreground='white',font=('arial',10,'bold'))
    detect_b.place(relx =0.79,rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25), (top.winfo_height()/2.25)))
        im = ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image = im
        label1.configure(text='')
        show_Detect_button(file_path)
    except:
        pass


upload = Button(top, text="Upload Image", command=upload_image, padx=10, pady=5)
upload.configure(background="#364156",foreground='white',font=('arial',20,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom', expand='True')
label1.pack(side='bottom', expand='True')
heading = Label(top,text='Nationality Detector',pady=20,font=('arial',25,'bold'))
heading.configure(background='#CDCDCD',foreground="#364156")
heading.pack()
top.mainloop()