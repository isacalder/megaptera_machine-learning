# Audio Analysis and Classification Project

## Overview

This project focuses on analyzing and classifying audio data, specifically whale songs, using machine learning techniques. The project processes audio files to extract features, create datasets, and apply various classification algorithms to predict the year associated with each audio file.

## Features

- **Audio Feature Extraction**: Extracts characteristics like duration, mean frequency, and Mel-frequency cepstral coefficients (MFCCs) from audio files.
- **Data Preprocessing**: Scales features and prepares datasets for training and testing.
- **Visualization**: Generates spectrograms, correlation matrices, and validation curves to analyze data and model performance.
- **Machine Learning Models**: Implements and compares multiple classification algorithms, including:
  - Gaussian Naive Bayes
  - Logistic Regression
  - Decision Trees
  - Random Forests
  - K-Nearest Neighbors
- **Prediction and Evaluation**: Predicts outcomes for test datasets and evaluates model accuracy.

## Data

The audio data used in this project was collected by a researcher working in Bahía Solano. These are private files intended for research purposes, and therefore, the original audio files are not shared.

The extracted features and processed datasets, however, are stored in CSV files (`audio_features_with_year.csv` and `audio_features_scaled.csv`) and are used throughout the analysis.

## Usage

1. **Dependencies**:
   Ensure the following libraries are installed:

   - `matplotlib`
   - `numpy`
   - `seaborn`
   - `pandas`
   - `librosa`
   - `scikit-learn`
   - `soundfile`

   Install them using pip if necessary:

   ```bash
   pip install matplotlib numpy seaborn pandas librosa scikit-learn soundfile
   ```

2. **Structure**:

   - Place all audio files in a directory named `Data`.
   - Ensure the filenames contain the year of recording (e.g., `Canto1_2019.WAV`).

3. **Running the Code**:

   - Execute the Python script or Jupyter Notebook to process audio files and generate datasets.
   - Use the extracted datasets to train and evaluate the classification models.

4. **Output**:

   - Feature datasets are saved as CSV files for further analysis.
   - Model predictions and accuracy scores are displayed and visualized.

## Results

The project includes:

- Comparative performance metrics for various classification models.
- Visualizations like spectrograms and correlation matrices to understand data and model behavior.

## Future Work

- Extend the dataset with additional labeled audio files.
- Experiment with advanced deep learning techniques for better feature extraction and classification.


