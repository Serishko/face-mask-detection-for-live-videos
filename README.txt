This project describes the basic concepts of how to make a model for face mask detection in today's scenario where all countries in the world are suffering from COVID-19. This project contains the following steps:

Step1: Make a Model
	-- First, we create a model from the datasets in the images and annotations folder. We extract the dimensions of the required face section we need, from .xml files and also the labels describing the section from these files. We apply those dimensions in the images with .png format to get only the face portion which acts as our features. After getting both the features and labels, we use them in order to train our model. We use deep learning in this project using keras since our datasets contain a huge amount of complex data. We use convolutional neural network to create our model since CNN is best for image processing.
	-- Refer to Face_mask_detection.ipynb for this step.

Step2: Saving the model and weights
	-- We then save our trained model in google drive and then to our local machine.
	-- Refer to face_mask detection model to view the saved model and its weights.

Step3: Using the model in live videos
	-- We have used haarcascade_frontalface_default.xml in order to detect the faces. Then, we sent the faces as input to our saved model to predict the labels. A rectangle box is drawn on the faces in the video with labels shown right above it.
	-- Refer to face_mask_detection_for_live_videos.pyw for this step.

Note: Press Esc in order to escape from the live videos

Datasets available from: https://www.kaggle.com/andrewmvd/face-mask-detection