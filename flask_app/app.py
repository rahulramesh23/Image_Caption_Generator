# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 12:42:36 2020

@author: rahul
"""
#from utils import test_image_caption,load_img
import utils
from flask import Flask, request, redirect, url_for, flash, jsonify
app = Flask(__name__)
model_file='trained_files/model-ep009-loss3.128-val_loss3.594.h5'
token_path='trained_files/tokenizer_new.pkl'
vocab_size=5438
max_length = 72

@app.route('/apitest/', methods=['GET','POST'])
def apitest():
    data = utils.load_img(request.files['file'], target_size=(299,299))
    desc = utils.test_image_caption(data)
    print('Result: ',desc)
    return desc

@app.route('/')
def index():
    return "<h1>Welcome to Image Caption Generator: Rahul</h1>"

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    utils.init(token_path,model_file, vocab_size, max_length)
    app.run(threaded=False, debug=False, host='0.0.0.0', port=5000)
