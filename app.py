import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load the saved models
model_paths = {
    "VGG16": "saved_models/model_2.h5",
    "VGG19": "saved_models/model_4.h5",
    #"ResNet101": "saved_models/model_7.h5"
}

models = {}
for model_name, model_path in model_paths.items():
    models[model_name] = tf.keras.models.load_model(model_path)

# Function to read images as array
def read_image(fn):
    image = Image.open(fn)
    return np.asarray(image.resize((120, 120)))

# Function to predict
def test_predict(image_array, model):
    result = model.predict(np.asarray([image_array]))

    itemindex = np.where(result==np.max(result))
    prediction = itemindex[1][0]
    prediction = class_names[prediction]
    return prediction

# Streamlit app
def main():
    st.title("Image Classification App")

    class_names = ['calling', 'clapping', 'cycling', 'dancing', 'drinking', 'eating',
                   'fighting', 'hugging', 'laughing', 'listening_to_music', 'running',
                   'sitting', 'sleeping', 'texting', 'using_laptop']
    
    # Dropdown to select the model
    selected_model = st.selectbox("Select Model", list(models.keys()))

    # Upload image through Streamlit UI
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.image(uploaded_image, caption="Uploaded Image.", use_column_width=True)

        # Read the uploaded image and convert to array
        img_array = read_image(uploaded_image)
        
        # Make prediction using the selected model
        prediction = test_predict(img_array, models[selected_model])
        
        # Display prediction results
        st.subheader(f"{selected_model} Prediction:")
        st.write("Predicted class:", prediction)
        
        # Display the uploaded image with the predicted class
        plt.imshow(img_array)
        plt.title(prediction)
        st.pyplot()

if __name__ == "__main__":
    main()
