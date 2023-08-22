
# Adapted from
# https://github.com/nachi-hebbar/Flower-Classification-Web-App-Streamlit/blob/main/Flower_Classification_WebApp%20(1).ipynb

import streamlit as st
import tensorflow as tf
import streamlit as st
from tensorflow.keras.applications.xception import Xception,preprocess_input as xception_preprocess_input

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('xception_model.keras')
    return model
with st.spinner('Model is being loaded..'):
    model=load_model()

st.write("""
         # Crochet Stitches Classification
         """
         )

file = st.file_uploader("Please upload an image of the crochet stitch", type=["jpg", "png", "jpeg"])
import cv2
from PIL import Image, ImageOps
import numpy as np
st.set_option('deprecation.showfileUploaderEncoding', False)
def import_and_predict(image_data, model):
    
    size = (180,180)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = xception_preprocess_input(image)

    img_reshape = img[np.newaxis,...]

    prediction = model.predict(img_reshape)

    return prediction
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, model)
    score = tf.nn.softmax(predictions[0])
    st.write(predictions)
    st.write(score)
    class_names = ['Single Crochet', 'Double Crochet', 'Half Double Crochet']
    res = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(class_names[np.argmax(score)], 100*np.max(score))
    st.text(res)
