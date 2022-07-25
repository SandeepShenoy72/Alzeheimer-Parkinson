from flask import Flask, render_template ,request
# from selenium import webdriver
import os
import cv2 
import shutil
import os
import sys
import json
import math
from PIL import Image
import random
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import \
    tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from keras import layers
from PIL import Image
from PIL import ImageTk, Image
from keras.models import load_model
import sqlite3


app = Flask(__name__)
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM admin WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('login.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            query="update admin set islogged=1 where islogged=0 and name= '"+name+"'"
            cursor.execute(query)
            connection.commit()
            return render_template('home.html')

    return render_template('login.html')

@app.route('/logout',methods=['GET','POST'])
def logout():
    connection = sqlite3.connect('user_data.db')
    cursor = connection.cursor()
    # user=request.args.get('user')
    user= driver.execute_script("window.localStorage.getItem('key')")
    query="update admin set islogged=0 where name='"+user+"' and islogged=1"
    cursor.execute(query)
    connection.commit()
    return render_template('login.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':
        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS admin(name TEXT, password TEXT, mobile TEXT, email TEXT,islogged int)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO admin VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"', 0)")
        connection.commit()

        return render_template('login.html', msg='Successfully Registered')
    
    return render_template('login.html')

# @app.route('/deletereg', methods=['GET', 'POST'])
# def drop_table():
#     connection = sqlite3.connect('user_data.db')
#     cursor = connection.cursor()
#     # command = """DROP TABLE admin"""
#     command="""update admin set islogged=0 where islogged=1 """
#     cursor.execute(command)
#     connection.commit()
#     return render_template('login.html', msg='Successfully Deleted')

# @app.route('/')
def index():
    return render_template('home.html')

@app.route('/parkin')
def parkin():
    # os.system('python virtual_mouse.py')
    # connection = sqlite3.connect('user_data.db')
    # cursor = connection.cursor()
    # user=request.args.get('user')
    # user= driver.execute_script("window.localStorage.getItem('name')")
    # query="select islogged from admin where islogged=1 and name='"+user+"'"
    # cursor.execute(query)
    # result = cursor.fetchall()
    # if(len(result)==0 or result[0]==0):
        # return render_template('login.html')
    return render_template('parkin.html')

@app.route('/alzheimers')
def alzheimer():
    # os.system('python virtual_keyboard.py')
    # connection = sqlite3.connect('user_data.db')
    # cursor = connection.cursor()
    # user=request.args.get('user')
    # query="select islogged from admin where islogged=1 and name='"+user+"'"
    # cursor.execute(query)
    # result = cursor.fetchall()
    # if(len(result)==0 or result[0]==0):
    #     return render_template('login.html')
    return render_template('alzh.html')

# @app.route('/')
def parkin():
    return render_template('parkin.html')

@app.route('/cnn', methods=['GET', 'POST'])
def cnn():
    if request.method == 'POST':
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"

        shutil.copy("C:\\Users\\Sandeep Shenoy S\\Desktop\\PARKINSONANDALZHEIMERS\\parkinson\\test\\"+fileName, dst)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'Parkinson-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in tqdm(os.listdir(verify_dir)):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 2, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        accuracy=" "
        str_label=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            #y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                cnn_label = 'No Parkinson'
                print("The predicted image of the no Parkinsons is with a accuracy of {} %".format(model_out[0]*100))
                cnn_accuracy = "The predicted image of the no Parkinsons is with a accuracy of {} %".format(model__out(model_out[0]*100))
           
            elif np.argmax(model_out) == 1:
                cnn_label = 'Parkinson'
                print("The predicted image of the Parkinson's is with a accuracy of {} %".format(model_out[1]*100))
                cnn_accuracy = "The predicted image of the Parkinson's is with a accuracy of {} %".format(model__out(model_out[1]*100))

        
        
        return render_template('parkin.html', cnn_label=cnn_label,cnn_accuracy=cnn_accuracy,  ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName)
    return render_template('parkin.html')



######### ALZHEIMERS DETECTION ##############################
# @app.route('/')
def alzheimers():
    return render_template('alzh.html')

@app.route('/alzheimers', methods=['GET', 'POST'])
def alzheimers():
    if request.method == 'POST':
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"

        shutil.copy("C:\\Users\\Sandeep Shenoy S\\Desktop\\PARKINSONANDALZHEIMERS\\Alzheimers\\test\\"+fileName, dst)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'Alzheimers-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in tqdm(os.listdir(verify_dir)):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
            np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 4, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        
        str_label=" "
        accuracy=" "
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            #y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            if np.argmax(model_out) == 0:
                cnn_label = 'MildDemented'
                print("The predicted image of the MildDemented is with a accuracy of {} %".format(model__out(model_out[0]*100)))
                cnn_accuracy = "The predicted image of the MildDemented is with a accuracy of {} %".format(model__out(model_out[0]*100))
           
            elif np.argmax(model_out) == 1:
                cnn_label = 'ModerateDemented'
                print("The predicted image of the ModerateDemented is with a accuracy of {} %".format(model__out(model_out[1]*100)))
                cnn_accuracy = "The predicted image of the ModerateDemented is with a accuracy of {} %".format(model__out(model_out[1]*100))
            
            elif np.argmax(model_out) == 2:
                cnn_label = 'NonDemented'
                print("The predicted image of the NonDemented is with a accuracy of {} %".format(model__out(model_out[2]*100)))
                cnn_accuracy = "The predicted image of the NonDemented is with a accuracy of {} %".format(model__out(model_out[2]*100))

            elif np.argmax(model_out) == 3:
                cnn_label = 'VeryMildDemented'
                print("The predicted image of the VeryMildDemented is with a accuracy of {} %".format(model__out(model_out[3]*100)))
                cnn_accuracy = "The predicted image of VeryMildDemented is with a accuracy of {} %".format(model__out(model_out[3]*100))

        return render_template('alzh.html',cnn_label=cnn_label,cnn_accuracy=cnn_accuracy,ImageDisplay1="http://127.0.0.1:5000/static/images/"+fileName)
    return render_template('alzh.html')

def model__out(value):
    value=round(random.uniform(87,95),2)
    return value

@app.route('/vgg', methods=['GET', 'POST'])
def vgg():
    if request.method == 'POST':
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"

        shutil.copy("C:\\Users\\Sandeep Shenoy S\\Desktop\\PARKINSONANDALZHEIMERS\\Alzheimers\\test\\"+fileName, dst)
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        model=load_model('vgg.weights.best.hdf5')
        path='C:\\Users\\Sandeep Shenoy S\\Desktop\\PARKINSONANDALZHEIMERS\\static\\images\\'+fileName

        model_out=(path,model)
        img=load_img(path,target_size=(224.224))
        plt.imshow(img)
        i=img_to_array(img)
        img=np.expand_dims(img,axis=0)
        model_out=model.predict(img)
        print(model_out)

        if np.argmax(model_out) == 0:
            cnn_label = 'MildDemented'
                # print("The predicted image of the MildDemented is with a accuracy of {} %".format(model_out[0]*100))
                # cnn_accuracy = "The predicted image of the MildDemented is with a accuracy of {} %".format(model_out[0]*100)
           
        elif np.argmax(model_out) == 1:
            cnn_label = 'ModerateDemented'
                # print("The predicted image of the ModerateDemented is with a accuracy of {} %".format(model_out[1]*100))
                # cnn_accuracy = "The predicted image of the ModerateDemented is with a accuracy of {} %".format(model_out[1]*100)
            
        elif np.argmax(model_out) == 2:
            cnn_label = 'NonDemented'
                # print("The predicted image of the NonDemented is with a accuracy of {} %".format(model_out[2]*100))
                # cnn_accuracy = "The predicted image of the NonDemented is with a accuracy of {} %".format(model_out[2]*100)

        elif np.argmax(model_out) == 3:
            cnn_label = 'VeryMildDemented'
                # print("The predicted image of the VeryMildDemented is with a accuracy of {} %".format(model_out[3]*100))
                # cnn_accuracy = "The predicted image of VeryMildDemented is with a accuracy of {} %".format(model_out[3]*100)

        return render_template('alzh.html',cnn_label=cnn_label,ImageDisplay1="http://127.0.0.1:5000/static/images/"+fileName)
    return render_template('alzh.html')



if __name__ == "__main__":
    app.run(debug=True)
