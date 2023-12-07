import streamlit as st
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.saving import load_model
import numpy as np
import cv2
from PIL import Image


st.title('Deep Learning Classifier App')
task = st.selectbox('Select Task', ['Choose one','Sentiment Classification', 'Tumor Detection'])

if task=='Tumor Detection':
    st.subheader('Tumor Detection with CNN')
    # CNN
    cnn_model = load_model("cnn_model.h5")

    img = st.file_uploader('Upload image', type=['jpeg', 'jpg', 'png'])

    def cnn_make_prediction(img,model):
        img=Image.open(img)
        img=img.resize((128,128))
        img=np.array(img)
        input_img = np.expand_dims(img, axis=0)
        res = model.predict(input_img)
        if res:
            return"Tumor Detected"
        else:
            return"No Tumor Detected"
                
    if img is not None:
        st.image(img, caption = "Image preview")
        if st.button('Submit'):
            pred = cnn_make_prediction(img, cnn_model)
            st.write(pred)


if task=='Sentiment Classification':
    arcs = ['Perceptron', 'Backpropagation', 'DNN', 'RNN', 'LSTM']
    arc = st.radio('Pick one:', arcs, horizontal=True)

    if arc == arcs[0]:
        # Perceptron
        with open("ppn_model.pkl",'rb') as file:
            perceptron = pickle.load(file)
        with open("ppn_tokeniser.pkl",'rb') as file:
            ppn_tokeniser = pickle.load(file)

        st.subheader('Movie Review Classification using Perceptron')
        inp = st.text_area('Enter message')
        
        def ppn_make_predictions(inp, model):
            encoded_inp = ppn_tokeniser.texts_to_sequences([inp])
            padded_inp = sequence.pad_sequences(encoded_inp, maxlen=500)
            res = model.predict(padded_inp)
            if res:
                return "Negative"
            else:
                return "Positive"       
        
        if st.button('Check'):
            pred = ppn_make_predictions([inp], perceptron)
            st.write(pred)

    elif arc == arcs[1]:
        # BackPropogation
        with open("bp_model.pkl",'rb') as file:
            backprop = pickle.load(file)
        with open("bp_tokeniser.pkl",'rb') as file:
            bp_tokeniser = pickle.load(file)

        st.subheader('Movie Review Classification using Backpropagation')
        inp = st.text_area('Enter message')
        
        def bp_make_predictions(inp, model):
            encoded_inp = bp_tokeniser.texts_to_sequences([inp])
            padded_inp = sequence.pad_sequences(encoded_inp, maxlen=500)
            res = model.predict(padded_inp)
            if res:
                return "Negative"
            else:
                return "Positive"        
        
        if st.button('Check'):
            pred = bp_make_predictions([inp], backprop)
            st.write(pred)


    elif arc == arcs[2]:
        # DNN
        dnn_model = load_model("dnn_model.h5")
        with open("dnn_tokeniser.pkl",'rb') as file:
            dnn_tokeniser = pickle.load(file)

        st.subheader('SMS Spam Classification using DNN')
        inp = st.text_area('Enter message')
        
        def dnn_make_predictions(inp, model):
            encoded_inp = dnn_tokeniser.texts_to_sequences(inp)
            padded_inp = sequence.pad_sequences(encoded_inp, maxlen=10, padding='post')
            res = (model.predict(padded_inp) > 0.5).astype("int32")
            if res:
                return "Spam"
            else:
                return "Ham"     
        
        if st.button('Check'):
            pred = dnn_make_predictions([inp], dnn_model)
            st.write(pred)


    elif arc == arcs[3]:
        # RNN 
        rnn_model = load_model("rnn_model.h5") 

        with open("rnn_tokeniser.pkl",'rb') as file:
            rnn_tokeniser = pickle.load(file)

        st.subheader('SMS Spam Classification using RNN')
        inp = st.text_area('Enter message')
        
        def rnn_make_predictions(inp, model):
            encoded_inp = rnn_tokeniser.texts_to_sequences(inp)
            padded_inp = sequence.pad_sequences(encoded_inp, maxlen=10, padding='post')
            res = (model.predict(padded_inp) > 0.5).astype("int32")
            if res:
                return "Spam"
            else:
                return "Ham"


        if st.button('Check'):
            pred = rnn_make_predictions([inp], rnn_model)
            st.write(pred)
    

    elif arc == arcs[4]:
        # LSTM
        lstm_model = load_model("lstm_model.h5") 

        with open("lstm_tokeniser.pkl",'rb') as file:
            lstm_tokeniser = pickle.load(file)

        st.subheader('Movie Review Classification using LSTM')
        inp = st.text_area('Enter message')

        def lstm_make_predictions(inp, model):
            inp = lstm_tokeniser.texts_to_sequences(inp)
            inp = sequence.pad_sequences(inp, maxlen=500)
            res = (model.predict(inp) > 0.5).astype("int32")
            if res:
                return "Negative"
            else:
                return "Positive"

        if st.button('Check'):
            pred = lstm_make_predictions([inp], lstm_model)
            st.write(pred)  