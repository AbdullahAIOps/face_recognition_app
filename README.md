# face_recognition_app
Gender Classification Using PCA and SVM
This project implements a gender classification system based on facial recognition. The system uses Principal Component Analysis (PCA) to extract the features of faces and Support Vector Machine (SVM) for classifying the gender of detected faces. The process involves detecting faces, extracting features, and then predicting the gender of each face in an image or video stream.

Project Features:
Face Detection: Using Haar Cascade Classifiers to detect faces in an image or video.

Feature Extraction: Implementing PCA to extract the most important features (eigenfaces) of faces.

Gender Prediction: Using a pre-trained SVM model to classify detected faces as male or female.

Real-time Processing: Supports both single image and video inputs for real-time gender prediction.

Model: A pre-trained SVM classifier with PCA-based feature extraction is used for gender classification.

Visualization: The system draws bounding boxes around detected faces and labels them with the predicted gender and confidence level.

Web Integration: Flask-based web application for uploading images and displaying gender prediction results.

Key Technologies:
Python: Main programming language used for the implementation.

OpenCV: Used for face detection and image manipulation.

PIL (Python Imaging Library): For image preprocessing.

Scikit-learn: For the SVM classifier.

Flask: Web framework for creating the user interface to upload images and show predictions.

Getting Started:
Clone this repository to your local machine.

Install the required libraries:

bash
Copy
Edit
pip install -r requirements.txt
Download the Haar Cascade for face detection and the pre-trained gender classification model (SVM + PCA).

Run the Flask app using:

bash
Copy
Edit
python app.py
Access the web interface to upload an image for gender classification.

Demo:
Upload an image and the system will automatically predict and display the gender (male/female) of detected faces.

Results include bounding boxes around the faces, eigenfaces, and confidence scores.

Future Enhancements:
Enhance the model for better accuracy using deep learning techniques like CNNs (Convolutional Neural Networks).

Add support for detecting and classifying multiple faces in real-time video streams.

Improve image resolution handling to minimize blurriness in eigenfaces.

License:
This project is licensed under the MIT License - see the LICENSE file for details.
