from flask import render_template, request
import os
import cv2
from app.faicel_recognization import process_image
from PIL import Image
import numpy as np
import matplotlib.image as mat_img
UPLOAD_FOLDER="static/uploads"
def index():
    return render_template("index.html")
def app():
    return render_template("app.html")
def gender():
    if request.method == "POST":
        f = request.files["image-name"]
        filename = f.filename
        path = os.path.join(UPLOAD_FOLDER,filename)
        f.save(path)
        # if img is a PIL image already:
        pred_img, predictions = process_image(path)
        prediction_filename = "prediction_img.jpg"
        cv2.imwrite(f"static/predict/{prediction_filename}", cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB))

        results = []
        for i, obj in enumerate(predictions):
            gray_img = obj["roi"]
            eigen_image = np.reshape(obj["eig_img"], (10, 5))

            gender_name = obj["prediction_name"]
            score = round(obj["score"]*100,2)
            gray_img_name = f"roi_{i}.jpg"
            eig_img_name = f"eig_roi{i}.jpg"

            mat_img.imsave(f"./static/predict/{gray_img_name}",gray_img,cmap="gray")
            mat_img.imsave(f"./static/predict/{eig_img_name}",eigen_image,cmap="gray")
            results.append((gray_img_name, eig_img_name, gender_name, score))

        return render_template("gender.html", fileuploaded=True, results = results)
    
    return render_template("gender.html")