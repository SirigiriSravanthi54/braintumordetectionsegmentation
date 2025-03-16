import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from skimage.transform import resize
import matplotlib.pyplot as plt
import pandas as pd
from databse import *
import re
import time
from datetime import datetime
import base64
import json
import plotly.graph_objects as go
import plotly.express as px

# Initialize database
init_db()

# Custom theme and styling
st.set_page_config(
    page_title="Brain Tumor Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)
# Custom metric functions for segmentation model
def dice_coef(y_true, y_pred):
    smooth = 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

def dice_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred)

# Session state initialization
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'verification_email' not in st.session_state:
    st.session_state.verification_email = None
if 'show_verification' not in st.session_state:
    st.session_state.show_verification = False

def verify_email_page():
    st.title("Email Verification")
    st.write("Please check your email for the verification code.")
    
    verification_code = st.text_input("Enter verification code:")
    if st.button("Verify Email"):
        if verify_email(st.session_state.verification_email, verification_code):
            st.success("Email verified successfully! Please login.")
            st.session_state.show_verification = False
            st.session_state.verification_email = None
        else:
            st.error("Invalid verification code")


def login_page():
    st.title("Login")
    
    if st.session_state.show_verification:
        verify_email_page()
        return
    
    tab1, tab2, tab3 = st.tabs(["Login", "Sign Up", "Forgot Password"])
    
    with tab1:
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")
        if st.button("Login"):
            is_valid, is_verified = verify_user(email, password)
            if is_valid:
                if is_verified:
                    st.session_state.logged_in = True
                    st.success("Logged in successfully!")
                    st.rerun()
                else:
                    st.error("Please verify your email first")
                    st.session_state.verification_email = email
                    st.session_state.show_verification = True
                    st.rerun()
            else:
                st.error("Invalid credentials")
#########################
    with tab2:
        new_email = st.text_input("Email", key="signup_email")
        new_password = st.text_input("Password", type="password", key="signup_password")
        confirm_password = st.text_input("Confirm Password", type="password")
        
        if st.button("Sign Up"):
            if not re.match(r"[^@]+@[^@]+\.[^@]+", new_email):
                st.error("Please enter a valid email address")
            elif new_password != confirm_password:
                st.error("Passwords do not match")
            elif len(new_password) < 6:
                st.error("Password must be at least 6 characters long")
            else:
                success, message = add_user(new_email, new_password)
                if success:
                    st.success("Account created! Please check your email for verification code.")
                    st.session_state.verification_email = new_email
                    st.session_state.show_verification = True
                    st.rerun()
                else:
                    st.error(f"Signup failed: {message}")
#####
    with tab3:
        if 'reset_email' not in st.session_state:
            st.session_state.reset_email = ""
        if 'reset_step' not in st.session_state:
            st.session_state.reset_step = 'email'

        if st.session_state.reset_step == 'email':
            reset_email = st.text_input("Email", key="reset_email_input")
            col1, col2 = st.columns([1,4])
            with col1:
                if st.button("Send Reset Code"):
                    if not re.match(r"[^@]+@[^@]+\.[^@]+", reset_email):
                        st.error("Please enter a valid email address")
                    else:
                        # Check if email exists in database
                        conn = sqlite3.connect('users.db')
                        c = conn.cursor()
                        c.execute("SELECT email FROM users WHERE email=?", (reset_email,))
                        user = c.fetchone()
                        conn.close()
                        
                        if user is None:
                            st.error("No account found with this email address")
                        else:
                            token = store_reset_token(reset_email)
                            if send_reset_email(reset_email, token):
                                st.session_state.reset_email = reset_email
                                st.session_state.reset_step = 'code'
                                st.success("Password reset code sent to your email!")
                                st.rerun()
                            else:
                                st.error("Failed to send reset email. Please try again later.")
                            
        else:  # reset_step == 'code'
            st.write(f"Enter the code sent to {st.session_state.reset_email}")
            reset_code = st.text_input("Reset Code", key="reset_code_input")
            new_password = st.text_input("New Password", type="password", key="new_password_input")
            confirm_password = st.text_input("Confirm New Password", type="password", key="confirm_password_input")
            
            col1, col2, col3 = st.columns([1,1,2])
            with col1:
                if st.button("Reset Password"):
                    if not reset_code:
                        st.error("Please enter the reset code")
                    elif len(new_password) < 6:
                        st.error("Password must be at least 6 characters long")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    elif verify_reset_token(st.session_state.reset_email, reset_code):
                        update_password(st.session_state.reset_email, new_password)
                        st.success("Password reset successful! Please login with your new password.")
                        st.session_state.reset_step = 'email'
                        st.session_state.reset_email = ""
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Invalid reset code")
            
            with col2:
                if st.button("Back"):
                    st.session_state.reset_step = 'email'
                    st.session_state.reset_email = ""
                    st.rerun()


def get_severity_info(prediction):
    severity_levels = {
        'No Tumor': {
            'level': 'Low',
            'description': 'No tumor detected. Regular check-ups recommended.',
            'color': 'green'
        },
        'Pituitary': {
            'level': 'Moderate',
            'description': 'Pituitary tumor detected. Usually benign but requires medical attention.',
            'color': 'yellow'
        },
        'Meningioma': {
            'level': 'Moderate to High',
            'description': 'Meningioma detected. Usually benign but location-dependent risks.',
            'color': 'orange'
        },
        'Glioma': {
            'level': 'High',
            'description': 'Glioma detected. Requires immediate medical attention.',
            'color': 'red'
        }
    }
    return severity_levels[prediction]
def preprocess_image_for_segmentation(image):
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = np.expand_dims(image, axis=[0, -1])
    return image
# Custom CSS
st.markdown("""
<style>
    .main {
        background-color: #262730;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 5px;
    }
    .stProgress > div > div > div > div {
        background-color: #4CAF50;
    }
    .status-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .report-box {
        background-color: #111111;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #111111;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .custom-tab {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .header-container {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 1rem;
        background-color: #262730;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# [Keep existing helper functions: dice_coef, dice_loss, etc.]

def create_analysis_report(prediction_type, results, image):
    """Create a downloadable analysis report"""
    report = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "analysis_type": prediction_type,
        "results": results,
    }
    return json.dumps(report, indent=4)

def plot_confidence_radar(predictions, labels):
    """Create a radar plot for confidence scores"""
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=predictions * 100,
        theta=labels,
        fill='toself',
        name='Confidence Scores'
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False
    )
    return fig

def create_download_link(file_name, content):
    """Create a download link for files"""
    b64 = base64.b64encode(content.encode()).decode()
    return f'<a href="data:file/txt;base64,{b64}" download="{file_name}">Download Report</a>'

def display_header():
    """Display the app header with user info"""
    st.markdown("""
        <div class="header-container">
            <h1>🧠 Brain Tumor Analysis Platform</h1>
            <div>
                <small>Last updated: {}</small>
            </div>
        </div>
    """.format(datetime.now().strftime("%Y-%m-%d")), unsafe_allow_html=True)
def integrated_analysis_app():
    st.markdown('<div class="custom-tab">', unsafe_allow_html=True)
    st.subheader("🔍 Brain Tumor Analysis")
    st.write("Upload a brain MRI image for classification and segmentation analysis")

    @st.cache_resource
    def load_models():
        classification_model = load_model('Bmodel.h5')
        segmentation_model = load_model('brain_tumor_segmentation.h5',
                                      custom_objects={'dice_coef': dice_coef,
                                                    'dice_loss': dice_loss})
        return classification_model, segmentation_model

    try:
        classification_model, segmentation_model = load_models()
        labels = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

        uploaded_file = st.file_uploader("Choose a brain MRI image...",
                                       type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Show loading progress
            with st.spinner('Processing image...'):
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)

            # Read image once and convert to format needed for both models
            uploaded_file.seek(0)
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            opencv_img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Convert BGR to RGB for classification
            rgb_img = cv2.cvtColor(opencv_img, cv2.COLOR_BGR2RGB)

            # Classification Analysis
            col1, col2 = st.columns([1, 1])
            
            # Display original image
            with col1:
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.subheader("Original Image")
                st.image(rgb_img, caption='Uploaded MRI Image', use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Prepare image for classification
            img_resized = resize(rgb_img, (224, 224, 3))
            img_resized = np.expand_dims(img_resized, axis=0)
            
            prediction = classification_model.predict(img_resized)
            predicted_class = np.argmax(prediction[0])
            confidence = prediction[0][predicted_class] * 100
            predicted_label = labels[predicted_class]
            
            severity_info = get_severity_info(predicted_label)
            
            # Display classification results
            with col2:
                st.markdown('<div class="report-box">', unsafe_allow_html=True)
                st.subheader("Classification Results")
                
                metric_col1, metric_col2 = st.columns(2)
                with metric_col1:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>Diagnosis</h3>
                            <h2>{predicted_label}</h2>
                        </div>
                    """, unsafe_allow_html=True)
                with metric_col2:
                    st.markdown(f"""
                        <div class="metric-card">
                            <h3>Confidence</h3>
                            <h2>{confidence:.1f}%</h2>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown(f"""
                    <div style='padding: 10px; background-color: {severity_info['color']}; 
                              border-radius: 5px; margin-top: 20px;'>
                        <h3 style='color: white;'>Severity Level: {severity_info['level']}</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.markdown(severity_info["description"])
                st.markdown('</div>', unsafe_allow_html=True)

            # Display classification visualization
            st.markdown('<div class="report-box">', unsafe_allow_html=True)
            st.subheader("Probability Distribution")
            chart_type = st.selectbox("Select visualization:",
                                    ["Bar Chart", "Radar Plot"])
            
            if chart_type == "Bar Chart":
                prob_df = pd.DataFrame({
                    'Tumor Type': labels,
                    'Probability': prediction[0] * 100
                })
                st.bar_chart(prob_df.set_index('Tumor Type'))
            else:
                radar_fig = plot_confidence_radar(prediction[0], labels)
                st.plotly_chart(radar_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # If tumor is detected, perform segmentation
            if predicted_label != 'No Tumor':
                st.markdown("---")
                st.subheader("🎯 Tumor Segmentation Analysis")
                
                # Perform segmentation using the already loaded image
                processed_image = preprocess_image_for_segmentation(opencv_img)
                seg_prediction = segmentation_model.predict(processed_image)
                seg_prediction = (seg_prediction > 0.5).astype(np.uint8)
                
                # Create overlay
                original_resized = cv2.resize(opencv_img, (256, 256))
                mask_3channel = np.zeros_like(original_resized)
                mask_3channel[:,:,1] = seg_prediction[0,:,:,0] * 255
                overlay = cv2.addWeighted(original_resized, 0.7, mask_3channel, 0.3, 0)
                
                # Convert overlay to RGB for display
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                
                # Display segmentation results
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.image(overlay_rgb, caption='Segmentation Result', use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Calculate and display metrics
                tumor_percentage = (np.sum(seg_prediction) / (256 * 256)) * 100
                severity = "High" if tumor_percentage > 10 else "Moderate" if tumor_percentage > 5 else "Low"
                
                with col2:
                    st.markdown('<div class="report-box">', unsafe_allow_html=True)
                    st.markdown('', unsafe_allow_html=True)

                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    with metric_col1:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>Tumor Coverage</h3>
                                <h2>{tumor_percentage:.2f}%</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col2:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>Detected Regions</h3>
                                <h2>{np.sum(seg_prediction > 0)}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with metric_col3:
                        st.markdown(f"""
                            <div class="metric-card">
                                <h3>Risk Level</h3>
                                <h2>{severity}</h2>
                            </div>
                        """, unsafe_allow_html=True)
                    st.markdown('', unsafe_allow_html=True)
                    
                    st.info("""
                    **Interpretation Guide:**
                    - Green overlay shows the detected tumor regions
                    - Brighter areas indicate higher confidence in tumor detection
                    - The percentage indicates the proportion of the image identified as tumor tissue
                    """)
                    st.markdown('</div>', unsafe_allow_html=True)

            # Create and offer combined report download
            report_data = {
                "classification": {
                    "predicted_class": predicted_label,
                    "confidence": float(confidence),
                    "severity_level": severity_info['level'],
                    "all_probabilities": {
                        label: float(prob) for label, prob in zip(labels, prediction[0])
                    }
                }
            }
            
            if predicted_label != 'No Tumor':
                report_data["segmentation"] = {
                    "tumor_coverage_percentage": float(tumor_percentage),
                    "detected_regions": int(np.sum(seg_prediction > 0)),
                    "risk_level": severity
                }
            
            st.markdown("---")
            # st.markdown('<div class="report-box">', unsafe_allow_html=True)
            if predicted_label != 'No Tumor':
                st.warning("⚠️ Please note that this is an AI prediction and should be confirmed by a medical professional.")
            else:
                st.success("✅ No tumor detected, but please consult with a medical professional for confirmation.")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # with col2:
            #     st.markdown('<div class="report-box">', unsafe_allow_html=True)
            #     st.markdown(create_download_link("analysis_report.json", report_data),
            #               unsafe_allow_html=True)
            #     st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("""Disclaimer: This application is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.""")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure all required model files are available in the directory.")

    st.markdown('</div>', unsafe_allow_html=True)

def main_app():
    display_header()
    integrated_analysis_app()

    # Sidebar
    st.sidebar.title("About")
    st.sidebar.info("""
    This app provides two types of brain tumor analysis:

    1. **Classification:**
    - Identifies tumor type
    - Provides severity assessment
    - Shows confidence scores
    - Types: Glioma, Meningioma, No Tumor, Pituitary

    2. **Segmentation:**
    - Localizes tumor regions
    - Shows tumor boundaries
    - Calculates tumor coverage
    - Provides visual overlay

    Upload your MRI scan to get started!
    """)

# Main app flow
if st.session_state.logged_in:
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.rerun()
    main_app()
else:
    login_page()
