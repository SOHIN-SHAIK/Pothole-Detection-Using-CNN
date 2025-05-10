from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import cv2
import random
import numpy as np
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import load_model
import pickle
import os


main = tkinter.Tk()
main.title("Pothole Detection System using Convolution Neural Networks")
main.geometry("1300x1200")

global filename
global model

labels = ['Plain','Pothhole']

def uploadDataset():
    global filename
    global labels
    labels = []
    filename = filedialog.askdirectory(initialdir=".")
    name = os.path.basename(filename)
    pathlabel.config(text=name)
    text.delete('1.0', END)
    text.insert(END,name+" loaded\n\n");
    

def trainCNN():
    global model
    text.delete('1.0', END)
    model = Sequential()
    model = load_model('sample.h5')
    text.insert(END,"CNN Model Generated")

    

def classifyFlower():
    filename = filedialog.askopenfilename(initialdir="sampleImages")
    image = cv2.imread(filename,0)
    img = cv2.resize(image, (100,100))
    im2arr = np.array(img)
    im2arr = im2arr.reshape(1,100,100,1)
    img = np.asarray(im2arr)
    #img = img.astype('float32')
    #img = img/255
    preds = model.predict(img)
    predict = np.argmax(preds)
    print(predict)
    img = cv2.imread(filename)
    img = cv2.resize(img, (400,400))
    if predict == 0:
        cv2.putText(img, 'No Pothole Detected', (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow('No Pothole Detected', img)
    else:
        cv2.putText(img, "Pothole Detected", (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (255, 0, 0), 2)
        cv2.imshow("Pothole Detected", img)
    cv2.waitKey(0)
    

def close():
    main.destroy()
    
    
font = ('times', 16, 'bold')
title = Label(main, text='Pothole Detection System using Convolution Neural Networks',anchor=W, justify=CENTER)
title.config(bg='yellow4', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Upload Pothole Dataset", command=uploadDataset)
upload.place(x=50,y=100)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='yellow4', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=150)

markovButton = Button(main, text="Train Dataset Using CNN", command=trainCNN)
markovButton.place(x=50,y=200)
markovButton.config(font=font1)

lexButton = Button(main, text="Upload Test Image & Detect Pothole", command=classifyFlower)
lexButton.place(x=50,y=250)
lexButton.config(font=font1)

predictButton = Button(main, text="Exit", command=close)
predictButton.place(x=50,y=300)
predictButton.config(font=font1)


font1 = ('times', 12, 'bold')
text=Text(main,height=15,width=78)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=450,y=100)
text.config(font=font1)


main.config(bg='magenta3')
main.mainloop()
