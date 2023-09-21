
import streamlit as st
import tensorflow as tf
import streamlit as st
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess

@st.cache(allow_output_mutation=True)
def load_model():
    model=tf.keras.models.load_model('models/final_model.keras')
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
    
    size = (224,224)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    
    # Convert images to grayscale
    image = rgb2gray(np.copy(image))

    # Reshape image from grayscale colorspace to rgb colorspace
    image = np.expand_dims(image, axis=3)
    image = image.repeat(3, axis=-1)
    
    # Data normalization
    img = resnet50_preprocess(image)

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
