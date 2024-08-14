import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras.models import load_model
import numpy as np 
import streamlit as st 


model=load_model(r'D:\computer vision practice code\all projects ml\image classification deep learning veg-fruit\models\image_classify.keras')

data_classs= ['apple',
 'banana',
 'beetroot',
 'bell pepper',
 'cabbage',
 'capsicum',
 'carrot',
 'cauliflower',
 'chilli pepper',
 'corn',
 'cucumber',
 'eggplant',
 'garlic',
 'ginger',
 'grapes',
 'jalepeno',
 'kiwi',
 'lemon',
 'lettuce',
 'mango',
 'onion',
 'orange',
 'paprika',
 'pear',
 'peas',
 'pineapple',
 'pomegranate',
 'potato',
 'raddish',
 'soy beans',
 'spinach',
 'sweetcorn',
 'sweetpotato',
 'tomato',
 'turnip',
 'watermelon']
st.header('Image Classification Model ')
# img = r'D:\computer vision practice code\all projects ml\image classification deep learning veg-fruit\Banana.jpg'
img = st.text_input('Enter the Image name','Apple.jpg')
img_width=180
img_height=180
img_load=tf.keras.utils.load_img(img,target_size=(img_width,img_height))
img_arr=tf.keras.utils.array_to_img(img_load)
img_batch=tf.expand_dims(img_arr,0)

predict=model.predict(img_batch)
score=tf.nn.softmax(predict)
st.image(img,width=200)

# st.write('veg/fruits is accuracy score  is  ',format(data_classs[np.argmax(score)]),np.max(score) *100)
st.write('veg/fruits is accuracy score  is  ',format(img),np.max(score) *100)