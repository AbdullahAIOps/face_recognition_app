import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import pickle

# ======================
# MODEL LOADING
# ======================
# Load face detector
haar = cv2.CascadeClassifier("/home/abdullah/Desktop/haarcascade_frontalface_default.xml")

# Load pre-trained gender model with verified class order
model_data = pickle.load(open("/home/abdullah/Desktop/verified_gender_model.pkl", "rb"))
model = model_data['model']          # SVM classifier
model_pca = model_data['pca']        # PCA model
mean = model_data['mean']            # Mean face
class_names = model_data['class_names']  # ['male', 'female']

font = cv2.FONT_HERSHEY_SIMPLEX

# ======================
# CORE PIPELINE FUNCTION
# ======================
def predict_gender(img, color="bgr"):
    """
    Process image and predict gender for all detected faces
    Returns:
        output_img: Image with bounding boxes and labels
        results: List of dicts with face data and predictions
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if color == "bgr" else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    faces = haar.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    output_img = img.copy()
    results = []

    for (x, y, w, h) in faces:
        # Preprocess the face
        roi = cv2.resize(gray[y:y+h, x:x+w], (100, 100)).astype(np.float32) / 255.0
        roi_normalized = roi.reshape(1, -1) - mean

        # PCA features
        eigen_features = model_pca.transform(roi_normalized)

        # Predict
        probabilities = model.predict_proba(eigen_features)[0]
        class_index = {label: i for i, label in enumerate(class_names)}
        prob_male = probabilities[class_index['male']]
        prob_female = probabilities[class_index['female']]

        # Decision logic
        CONF_THRESH = 0.65
        if prob_male >= CONF_THRESH and prob_male > prob_female:
            gender, confidence = "male", prob_male
            box_color = (0, 255, 0)
        elif prob_female >= CONF_THRESH:
            gender, confidence = "female", prob_female
            box_color = (0, 255, 0)
        else:
            gender = "male" if prob_male > prob_female else "female"
            confidence = max(prob_male, prob_female)
            box_color = (0, 0, 255)

        # Draw on image
        cv2.rectangle(output_img, (x, y), (x+w, y+h), box_color, 2)
        cv2.putText(output_img, f"{gender} {confidence:.0%}", (x, y-10), font, 0.9, box_color, 2)

        # Save results for template
        results.append({
            "roi": roi,
            "eig_img": eigen_features[0],
            "prediction_name": gender,
            "score": confidence
        })

    return output_img, results


# ======================
# UTILITY FUNCTIONS
# ======================
def process_image(image_path):
    img = np.array(Image.open(image_path))
    processed_img, predictions = predict_gender(img)
    return processed_img, predictions


def process_video(video_path):
    """Process video stream"""
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        processed_frame, _ = predict_gender(frame, "bgr")
        cv2.imshow("Gender Detection", processed_frame)
        
        if cv2.waitKey(1) == ord("q"):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# ======================
# EXAMPLE USAGE
# ======================
if __name__ == "__main__":
    # 1. Test single image
    process_image("/home/abdullah/Desktop/female_face/female41.png")
    
    # 2. Process video
    #process_video("/home/abdullah/Downloads/captian.mp4")
    
    # 3. Debug prediction on specific image
    debug_image = np.array(Image.open("test_face.jpg"))
    result, features = predict_gender(debug_image)
    
    if features is not None:
        print("Feature vector shape:", features.shape)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()

