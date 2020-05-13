# Image caption generator 

A deep learning model that combines the use of convolutional neural networks and recurrent neural networks for describing photo. The goal is to create a REST API using the trained model. The Flask app must accept a test image and generate a sentence describing the image. 

Dataset: [Flicker 30K](https://www.kaggle.com/hsankesara/flickr-image-dataset) containing roughly 31k images with 5 caption per image.

Usage: 
1. Run the colab notebook to generate a trained model
2. Test using the test script
3. Edit 'app.py' according to your final model configuration
4. Run app.py to launch app on local host
5. Test the app for test image using curl:
   curl -F "file=@D:/Projects/Image_caption_sys/flickr30k_images/flickr30k_images/438106.jpg" 127.0.0.1:5000/apitest/
6. Additionally you can containarize this app using Docker or serve using Heroku for deployment testing    

References: 
[Deep Learning Photo Caption Generator](https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/)
[Your flask app on heroku](https://stackabuse.com/deploying-a-flask-application-to-heroku/)
