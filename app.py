import streamlit as st
import tensorflow as tf
from PIL import Image  # Use PIL for image processing
import pandas as pd
st.set_page_config(page_title="Food Vision Predictor", page_icon="", layout="centered")

st.header("Food Vision Milestone Project")
st.subheader("This project aims to classify food items into various categories using computer vision techniques.")

SUPPORTED_IMAGE_TYPES = ("jpeg", "jpg", "png")
uploaded_file = st.file_uploader(label="Upload your food image", type=SUPPORTED_IMAGE_TYPES)

class_names = ['apple_6',
 'apple_braeburn_1',
 'apple_crimson_snow_1',
 'apple_golden_1',
 'apple_golden_2',
 'apple_golden_3',
 'apple_granny_smith_1',
 'apple_hit_1',
 'apple_pink_lady_1',
 'apple_red_1',
 'apple_red_2',
 'apple_red_3',
 'apple_red_delicios_1',
 'apple_red_yellow_1',
 'apple_rotten_1',
 'cabbage_white_1',
 'carrot_1',
 'cucumber_1',
 'cucumber_3',
 'eggplant_violet_1',
 'pear_1',
 'pear_3',
 'zucchini_1',
 'zucchini_dark_1'] 

with st.expander("See class names"):
    st.table(class_names)

@st.cache_resource
def load_model():
    try:
        # Handle potential errors during model loading
        return tf.keras.models.load_model("resnet/content/MOdel_RESNET50/")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None  # Return None to indicate an error

# Load the model
model = load_model()


def preprocess_image(image):
    try:
        # Assuming your model expects a specific input shape, resize if needed
        # (replace with your actual resizing logic based on model requirements)
        img = image.resize((224, 224))  # Example resize to 224x224
        # Convert to a NumPy array and expand dims for model input
        img_array = tf.expand_dims(tf.convert_to_tensor(img), axis=0)
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {e}")
        return None  # Return None to indicate an error

if uploaded_file is not None:
    file = Image.open(uploaded_file)
    # Use PIL's open function to handle different image formats
    img = preprocess_image(image=file)

    if img is not None:
        prediction = model.predict(img)
        with st.expander("see prediction result"):
            df = pd.DataFrame({
    "Class Name": class_names,
    "Prediction Probability": prediction[0]
        })
            st.dataframe(df,width=2000,hide_index=True)
        
        if tf.reduce_max(prediction[0])>0.5:
            try:
                predicted_class = tf.argmax(prediction[0]).numpy()
                st.write(f"this is {class_names[predicted_class]}")
            except:
                raise TypeError
        
        else:
            st.error("This image may not seems to belong to above mentioned class. Please reload the relevant file")
                
         # Replace with your actual class names
        
        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
        
        
st.divider()

