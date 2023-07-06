
import os
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.utils import img_to_array 
import numpy as np
from keras.preprocessing import image
import tensorflow as tf  
from keras.utils import load_img, img_to_array 
from skimage import io
from werkzeug.utils import secure_filename
from keras.models import load_model

app = Flask(__name__)


# Loading model 
model = load_model('depression (1).h5') 

# Function to preprocess the uploaded image
def preprocess_image(img_path):
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    img_array =  tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array
    


def resize_images( X_in ):
    imgs=[]
    for path in X_in :
        img =io.imread(path)


# Function to make predictions using the loaded model
def predict_label(img_array):
    # Preprocess the image array
    img_array = preprocess_image(img_array)

    # Make predictions
    predictions = model.predict(img_array)
    label = np.argmax(predictions)

    return label

    

@app.route('/predict', methods=['POST'])
def upload_predict():
        img_path= upload_image()

        # Predict the label for the image
        label = predict_label(img_path)

        # Define your own labels here based on your model's output
        labels = {
            0: 'depressed',
            1: 'non_depressed',
            # Add more labels as needed
        }

        # Get the label name based on the predicted index
        predicted_label = labels.get(label, 'Unknown')
        # Remove the temporary image file
        os.remove(img_path)

        return jsonify({'predicted_label': predicted_label,})


@app.route("/upload-image", methods=["POST"])
def upload_image():
    print(request.files)
    if "image" not in request.files:
        return "No image file found"
    image = request.files["image"]
    print(image)
    if image.filename == "":
        return "No image selected"
    if image:
        filename = secure_filename(image.filename)
        # image.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        dbPath = os.path.join(r"uploads", filename) 
        # Save dbPath to database
        image.save(dbPath)
        return dbPath

if __name__ == '__main__':
    app.run(debug=True)
