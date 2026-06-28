import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Page configuration
st.set_page_config(
    page_title="Fruit Ripeness Classification",
    page_icon="🍎",
    layout="wide"
)

# Custom CSS for green and white theme
st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
    }
    .main-title {
        color: #2E8B57;
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
    }
    .description {
        color: #555555;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin-top: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# Class labels
CLASS_LABELS = [
    'Fresh Apple',
    'Fresh Banana',
    'Fresh Orange',
    'Rotten Apple',
    'Rotten Banana',
    'Rotten Orange'
]

def load_model():
    """
    Load the pre-trained MobileNetV2 model.
    Returns the loaded model.
    """
    try:
        model = tf.keras.models.load_model('mobilenet_model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    """
    Preprocess the uploaded image for prediction.
    Resizes to 224x224 and normalizes pixel values.
    
    Args:
        image: PIL Image object
        
    Returns:
        Preprocessed numpy array
    """
    # Resize image to 224x224
    image = image.resize((224, 224))
    
    # Convert to numpy array
    image_array = np.array(image)
    
    # Normalize pixel values to [0, 1]
    image_array = image_array / 255.0
    
    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

def predict_image(model, image_array):
    """
    Make prediction on the preprocessed image.
    
    Args:
        model: Loaded TensorFlow model
        image_array: Preprocessed image array
        
    Returns:
        Predicted class index and probabilities
    """
    predictions = model.predict(image_array)
    predicted_class = np.argmax(predictions[0])
    probabilities = predictions[0]
    
    return predicted_class, probabilities

def display_probabilities(probabilities):
    """
    Display prediction probabilities as a horizontal bar chart.
    
    Args:
        probabilities: Array of prediction probabilities
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Create horizontal bar chart
    y_pos = np.arange(len(CLASS_LABELS))
    colors = ['#90EE90' if prob == max(probabilities) else '#87CEEB' for prob in probabilities]
    
    ax.barh(y_pos, probabilities, color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(CLASS_LABELS)
    ax.set_xlabel('Probability', fontsize=12)
    ax.set_title('Prediction Probabilities for All Classes', fontsize=14, color='#2E8B57')
    ax.set_xlim(0, 1)
    
    # Add probability values to bars
    for i, v in enumerate(probabilities):
        ax.text(v + 0.01, i, f'{v:.2%}', va='center', fontsize=10)
    
    plt.tight_layout()
    st.pyplot(fig)

def main():
    """
    Main function to run the Streamlit application.
    """
    # Load model at startup
    model = load_model()
    
    if model is None:
        st.error("Failed to load the model. Please ensure 'mobilenet_model.h5' is in the same directory.")
        return
    
    # Sidebar with project information
    with st.sidebar:
        st.markdown("## 📊 Project Information")
        st.markdown("---")
        st.markdown("**Model Name:** MobileNetV2")
        st.markdown("**Dataset:** Fruits Fresh and Rotten")
        st.markdown("**Number of Classes:** 6")
        st.markdown("**Author:** [Your Name]")
        st.markdown("---")
        st.markdown("### Class Labels:")
        for label in CLASS_LABELS:
            st.markdown(f"- {label}")
    
    # Main content area
    st.markdown('<h1 class="main-title">Fruit Ripeness Classification using MobileNetV2</h1>', 
                unsafe_allow_html=True)
    
    st.markdown('<p class="description">Upload a fruit image to classify it as fresh or rotten. '
                'Supported classes: Apple, Banana, and Orange.</p>', 
                unsafe_allow_html=True)
    
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=['jpg', 'jpeg', 'png'],
        help="Upload a JPG, JPEG, or PNG image of a fruit"
    )
    
    if uploaded_file is not None:
        try:
            # Display uploaded image
            image = Image.open(uploaded_file)
            
            # Create two columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("### Uploaded Image")
                st.image(image, use_container_width=True)
            
            # Preprocess and predict
            image_array = preprocess_image(image)
            predicted_class, probabilities = predict_image(model, image_array)
            
            with col2:
                st.markdown("### Prediction Results")
                
                # Display predicted class
                predicted_label = CLASS_LABELS[predicted_class]
                confidence = probabilities[predicted_class] * 100
                
                st.markdown(f"**Predicted Class:** {predicted_label}")
                st.markdown(f"**Confidence:** {confidence:.2f}%")
                
                # Success message
                st.markdown('<div class="success-message">✅ Prediction completed successfully!</div>', 
                           unsafe_allow_html=True)
            
            # Display probabilities for all classes
            st.markdown("---")
            st.markdown("### Prediction Probabilities")
            display_probabilities(probabilities)
            
            # Display detailed probabilities table
            st.markdown("### Detailed Probabilities")
            prob_data = []
            for i, label in enumerate(CLASS_LABELS):
                prob_data.append({
                    'Class': label,
                    'Probability': f'{probabilities[i]:.4f}',
                    'Percentage': f'{probabilities[i] * 100:.2f}%'
                })
            
            st.table(prob_data)
            
        except Exception as e:
            st.error(f"Error processing image: {e}")
            st.error("Please ensure the uploaded file is a valid image.")
    
    else:
        # Display instructions when no file is uploaded
        st.info("👆 Upload an image to get started with the classification.")

if __name__ == "__main__":
    main()
