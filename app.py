import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import requests
import tempfile
import os

# Custom Depthwise Convolution function
def custom_depthwise_conv2d(*args, **kwargs):
    kwargs.pop('groups', None)  # Remove 'groups' if present in kwargs
    return tf.keras.layers.DepthwiseConv2D(*args, **kwargs)

def page1():
    # Load the model from the embedded URL
    @st.cache_resource
    def load_custom_model():
        model_url = "https://storage.googleapis.com/streamlit-bucket2024/keras_model.h5"
        temp_model_path = os.path.join(tempfile.gettempdir(), 'coffee_model.h5')

        response = requests.get(model_url)
        with open(temp_model_path, 'wb') as f:
            f.write(response.content)

        model = load_model(temp_model_path, custom_objects={'DepthwiseConv2D': custom_depthwise_conv2d})
        return model

    def load_labels():
        labels_url = "https://storage.googleapis.com/streamlit-bucket2024/labels.txt"
        response = requests.get(labels_url)
        class_names = response.text.splitlines()
        return class_names

    def predict(image, model, class_names):
        # Convert image to RGB if it has an alpha channel
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = image.resize((224, 224))
        image_array = np.asarray(image)
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
        data[0] = normalized_image_array

        prediction = model.predict(data)
        return prediction

    st.markdown("<h1 style='text-align: center;'>Animal Classifier</h1>", unsafe_allow_html=True)

    model = load_custom_model()
    class_names = load_labels()


    col1, col2 = st.columns(2)

    with col1:
        # Toggle between uploading an image and taking a picture
        mode = st.radio("Select Mode", ["Upload Image", "Take a Picture"])

        uploaded_file = None
        camera_file = None
        class_name = ""
        confidence_score = 0.0

        if mode == "Upload Image":
            # Upload image (supports both PNG and JPG)
            uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image.', use_column_width=True)

                prediction = predict(image, model, class_names)
                index = np.argmax(prediction)
                class_name = class_names[index].strip()
                confidence_score = prediction[0][index]

                st.success("Image uploaded successfully!")
            else:
                st.warning("Please upload an image to proceed.")

        else:
            camera_file = st.camera_input("Take a picture")
            if camera_file is not None:
                image = Image.open(camera_file)
                st.image(image, caption='Captured Image.', use_column_width=True)

                prediction = predict(image, model, class_names)
                index = np.argmax(prediction)
                class_name = class_names[index].strip()
                confidence_score = prediction[0][index]

                st.success("Picture captured successfully!")
            else:
                st.warning("Please take a picture to proceed.")

    with col2:
        st.header("Prediction Result")
        if (mode == "Upload Image" and uploaded_file is not None) or (mode == "Take a Picture" and camera_file is not None):
            st.write(f"Class: {class_name[2:]}")  # Display class name starting from the third character
            st.write(f"Confidence: {confidence_score * 100:.2f}%")  # Display as percentage
        else:
            st.write("Please upload an image or take a picture to see the prediction.")

def page2():
    import streamlit as st

    # Title for the "About Us" page
    st.title("About Us")

    # Introduction section
    st.header("Our Mission")
    st.write("""
    Welcome to [Your Project/Company Name]! We are committed to [insert core mission or values, e.g., providing innovative AI solutions, creating user-friendly educational tools, etc.]. 
    Our goal is to empower our users with tools that make [specific domain or tasks] more efficient and impactful.
    """)

    # Team section
    st.header("Our Team")
    st.write("""
    We are a diverse team of passionate professionals who specialize in [list key specialties: machine learning, data science, software development, etc.].
    Together, we bring a unique blend of expertise and creativity to every project we undertake.
    """)

    # Values section
    st.header("Our Values")
    st.write("""
    1. **Innovation**: We believe in continuous improvement and thinking outside the box.
    2. **Integrity**: We operate with transparency and hold ourselves to high ethical standards.
    3. **Collaboration**: We value teamwork and strive to build lasting relationships with our clients and partners.
    4. **User-Centric Design**: Our solutions are designed to be intuitive and practical for our users.
    """)

    # Contact Information
    st.header("Contact Us")
    st.write("""
    Feel free to reach out for collaborations, questions, or more information:
    - **Email**: contact@yourprojectname.com
    - **Phone**: +1 (123) 456-7890
    - **Location**: [City, Country]
    """)

    # Additional customization
    st.write("Thank you for visiting our 'About Us' page! We look forward to connecting with you.")

    



st.set_page_config(page_title="Animal Classifier", page_icon=":smiley:")

# Add a sidebar to navigate between pages
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Animal Classifier", "About Us"])

if page == "Animal Classifier":
    page1()
else:
    page2()
