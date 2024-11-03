# Sanket - ISL

## Why Sanket?
According to WHO, approximately 466 million people worldwide have disabling hearing loss, with over 6 million deaf and hard of hearing individuals in India alone. Despite sign language being a crucial mode of communication for this community, there exists a significant communication gap between signers and non-signers. Sanket addresses this challenge by providing a real-time sign language recognition system that converts Indian Sign Language (ISL) gestures into text, facilitating seamless communication between the deaf community and the general public. With support for 262 commonly used signs, Sanket represents a significant step toward breaking down communication barriers and promoting inclusivity in daily interactions.

## Project Overview
This project is Indian a sign language recognition system that processes sign language gestures and converts them into text. The system includes components for landmark extraction, model training, inference, visualization, and a web interface for real-time recognition.

## Repository Structure

### Core Components
- **Sanket/** - Web interface implementation
  - Features real-time inference using webcam
  - Sample video testing capabilities
  - Sentence formation using GenAI
  - Interactive user interface for sign language recognition

### Notebooks and Scripts
- **INCLUDE - Landmark Extraction.ipynb**
  - Landmark extraction and processing pipeline
  - Specifically designed for the ISL (Indian Sign Language) dataset
  - Saves extracted landmarks for further processing

- **sanket_model.ipynb**
  - Complete model training pipeline
  - Data loading and preprocessing
  - Training process implementation
  - Performance visualization and results analysis
  - Model evaluation graphs and metrics

- **Sanket Model Testing.ipynb**
  - Testing framework for the trained model
  - Supports testing on:
    - User-provided videos
    - Dataset videos
  - Performance evaluation tools

- **landmark_visualization.ipynb**
  - Visualization tools for extracted landmarks
  - Video saving capabilities with landmark overlay
  - Debug and analysis functionality

- **Sanket Inference.py**
  - Standalone inference script
  - Manual inference implementation
  - Optimized for production use

## Setup and Usage

1. Clone the repository
```bash
git clone https://github.com/AbhishekNair050/Sanket-ISL/
```

2. Navigate to the Sanket directory
```bash
cd Sanket
```

3. Run the application
```bash
python main.py
```

## Features
- Real-time sign language recognition
- Support for video file processing
- Landmark visualization tools
- Web-based interface
- GenAI-powered sentence formation
- Comprehensive model testing capabilities

## Dataset
The project utilizes the INCLUDE dataset, which contains 262 distinct sign language labels. This comprehensive dataset provides a robust foundation for training and evaluating the model across a wide range of Indian Sign Language gestures.

## Model Performance

### Overall Metrics
- Training Accuracy: 0.89
- Validation Accuracy: 0.7408
- Validation Loss: 1.7784
- Test Accuracy: 0.7395
- Test Loss: 1.7733
- Top-3 Accuracy: 0.8128 (Validation Set)

### Performance Analysis
- The model demonstrates strong performance with an 89% training accuracy and maintains consistent performance on unseen data with approximately 74% accuracy on both validation and test sets.
- The top-3 accuracy of 81.28% indicates high reliability in capturing the correct gesture within the top three predictions, making it particularly suitable for real-time applications.
- F1 scores show balanced performance across all gesture categories, indicating no bias toward specific classes.

### Areas for Improvement
- Analysis of misclassifications revealed that errors primarily occur between visually similar gestures.
- Most confusion occurs between gestures sharing overlapping visual or contextual features.
- Future improvements could focus on enhancing the model's ability to distinguish between subtle differences in similar gestures.
