# 🧠 Brain Tumor Classification Using Deep Learning
*A Streamlit-based AI Tool for Medical Image Diagnosis*

## 📌 Description
This project is a deep learning-powered web application for classifying brain MRI/CT images as **Healthy** or **Tumor**. It uses a Convolutional Neural Network (CNN) trained on medical images and provides an intuitive Streamlit interface with support for batch classification, contrast enhancement, and downloadable diagnostic reports in PDF format.

---

## 🚀 Features
- 🧠 Classifies brain images using a trained CNN model
- 📊 Displays prediction confidence and class probabilities
- 🖼️ Shows original vs. preprocessed images
- 🧾 Auto-generates PDF reports with metadata, histograms, and prediction summaries
- 📁 Batch mode: Upload a ZIP of images and download CSV results
- 📚 View prediction history and model architecture summary

---

## 🖥️ Demo Screenshots
 
 # Dashboard 
 [![Dashboard Screenshot](assets/dashboard_1.png)](assets/dashboard_1.png)
 [![Dashboard Screenshot](assets/dashboard_2.png)](assets/dashboard_2.png)
 [![Dashboard Screenshot](assets/dashboard_3.png)](assets/dashboard_3.png)

 # Classification Result
 [![Result Screenshot](assets/prediction_result_1_1.1.png)](assets/prediction_result_1_1.1.png)
 [![Result Screenshot](assets/prediction_result_1_1.2.png)](assets/prediction_result_1_1.2.png)
 [![Result Screenshot](assets/prediction_result_1_1.3.png)](assets/prediction_result_1_1.3.png)
 [![Result Screenshot](assets/prediction_result_1_1.4.png)](assets/prediction_result_1_1.4.png)
 [![Result Screenshot](assets/prediction_result_1_1.5.png)](assets/prediction_result_1_1.5.png)
 [![Result Screenshot](assets/prediction_result_1_1.6.png)](assets/prediction_result_1_1.6.png)
 [![Result Screenshot](assets/prediction_result_1_1.7.png)](assets/prediction_result_1_1.7.png)
 [![Result Screenshot](assets/prediction_result_2.png)](assets/prediction_result_2.png)
 [![Result Screenshot](assets/prediction_result_3.png)](assets/prediction_result_3.png)

 # Generated File
 [![Fie Screenshot](assets/file_preview_1.1.png)](assets/file_preview_1.1.png)
 [![Fie Screenshot](assets/file_preview_1.2.png)](assets/file_preview_1.2.png)
 [![Fie Screenshot](assets/file_preview_1.3.png)](assets/file_preview_1.3.png)
 [![Fie Screenshot](assets/file_preview_1.4.png)](assets/file_preview_1.4.png)
 [![Fie Screenshot](assets/file_preview_1.5.png)](assets/file_preview_1.5.png)
 [![Fie Screenshot](assets/file_preview_2.png)](assets/file_preview_2.png)
 
---

## 🧪 Tech Stack & Libraries

- `Python 3.10`
- `TensorFlow / Keras`
- `Streamlit`
- `Pillow (PIL)`
- `Matplotlib`, `Seaborn`
- `FPDF`
- `NumPy`, `Pandas`
- `OpenCV`, `Scikit-learn`

---

## 🎯 Future Improvements

Multiclass tumor detection (glioma, meningioma, pituitary)

DICOM format support

Cloud deployment (e.g., Hugging Face, Streamlit Cloud)

Heatmap overlays (Grad-CAM)

Patient form integration and database storage

---

## 📦 Model Not Included 
To run the app, download the trained model file from:  
[Google Drive Link](https://drive.google.com/file/d/1yshagIhfq15iDHo_0-3SRw33lavghMiT/view?usp=sharing)

Then place it in the project directory as `brain_tumor_model.h5`.
