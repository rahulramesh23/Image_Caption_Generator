# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:12:01 2020

@author: rahul
"""
from pickle import load
from numpy import argmax
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
#model_file='model_files/model-ep004-loss3.441-val_loss3.820.h5'


# extract features from each photo in the directory
def extract_features(model,image):
    # load the model
    #model = InceptionV3()
    #model = InceptionV3(weights='imagenet')

    # re-structure the model
    #model= Model(model.input, model.layers[-2].output)

    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    image = preprocess_input(image)
    # get features
    feature = model.predict(image, verbose=0)
    return feature

# map an integer to a word
def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate a description for an image
def generate_desc2(model, tokenizer, photo, max_length):
	# seed the generation process
	in_text = 'startseq'
	# iterate over the whole length of the sequence
	for i in range(max_length):
		# integer encode input sequence
		sequence = tokenizer.texts_to_sequences([in_text])[0]
		# pad input
		sequence = pad_sequences([sequence], maxlen=max_length)
		# predict next word
		yhat = model.predict([photo,sequence], verbose=0)
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
	return in_text

def generate_desc(model_img_caption,tokenizer, photo, max_length):
    # load the model for image caption generator
    #model_file='model_files/model-ep009-loss3.128-val_loss3.594.h5'
    #model_img_caption = load_model(model_file)
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for i in range(max_length):
                # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        #predict next word
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

def test_image_caption(m1,m2,photo_path,max_length=72):
    # load the tokenizer
    tokenizer = load(open('tokenizer_new.pkl', 'rb'))
    # pre-define the max sequence length (from training)
    #max_length = 72
    # load the model
    #model_file='model_files/model-ep004-loss3.441-val_loss3.820.h5'
    #model = load_model(model_file)
    # load and prepare the photograph
    	# load the photo
    image = load_img(photo_path, target_size=(299,299))
    photo = extract_features(m1,image)
    #generate description
    description = generate_desc(m2,tokenizer, photo, max_length)
    print(description)
    return description

def read_csv_pd(filename):
    doc = pd.read_csv(filename,delimiter = '|')
    doc.columns = ['image_name', 'comment_number', 'comment']
    idx=doc.comment.index[doc.comment.isnull()]
    doc.comment[idx]=doc.comment[idx-1]
    doc.comment_number[idx]=doc.comment_number[idx-1]
    return doc

