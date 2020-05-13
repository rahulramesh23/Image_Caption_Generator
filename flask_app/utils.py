# -*- coding: utf-8 -*-
"""
Created on Sat May  9 00:00:51 2020

@author: rahul
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:42:36 2020

@author: rahul
"""
from pickle import load
from numpy import argmax
from PIL import Image
from tensorflow import get_default_graph
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import numpy as np
import pickle as p
import json
global graph
# extract features from each photo in the directory
# define the captioning model
def define_model(vocab_size, max_length):
    # feature extractor model
    inputs1 = Input(shape=(2048,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    #print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model

def extract_features(img):
    #global model_x
    image = img_to_array(img)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)

    #model = InceptionV3(weights='imagenet')
    # re-structure the model
    #model= Model(model.input, model.layers[-2].output)
    #with graph.as_default():
    feature = model_x.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None


# generate a description for an image
def generate_desc(tokenizer, photo, max_length=72,vocab_size=5438):
    # load the model for image caption generator
    #model_file='model-ep009-loss3.128-val_loss3.594.h5'
    #model_img_caption = define_model(vocab_size, max_length=72)
    #model_img_caption.load_weights(model_file)
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        #predict next word
        #with graph.as_default():
        yhat = model_img_caption.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        if word is None:
            break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

def test_image_caption(photo_path,max_length=72):
    # pre-define the max sequence length (from training)
    max_length = 72

    # load and prepare the photograph
    print('extracting features of the photo using InceptionV3')
    photo = extract_features(photo_path)
    # generate description
    print('Predicting the Caption')
    description = generate_desc(tokenizer, photo, max_length)
    #print(description)
    return description

def init(token_path,model_file,vocab_size,max_length):
    # load the tokenizer
    global tokenizer
    tokenizer = load(open(token_path, 'rb'))
    print('Initialising the model')
    global model_x,model_img_caption
    model_x = InceptionV3(weights='imagenet')
    # re-structure the model
    model_x= Model(model_x.input, model_x.layers[-2].output)
    print('Inception model loaded successfully')

    model_img_caption = define_model(vocab_size, max_length=72)
    model_img_caption.load_weights(model_file)
    print('Caption generator loaded successfully')
    #global graph
    #graph = get_default_graph() 
    return 1
