# Audio Analysis and Classification Project

## Overview
This project implements an end-to-end pipeline for analyzing and classifying audio recordings using machine learning techniques. The pipeline extracts audio features, preprocesses the data, trains various classification models, and evaluates their performance. The primary use case for this project is the classification of whale songs by year.

---

## Features
1. **Audio Analysis**
   - Reads audio files in `.wav` format.
   - Computes spectrograms (STFT and Mel-spectrogram).
   - Extracts statistical and spectral features such as MFCCs, Mel-spectrogram values, duration, and magnitudes.

2. **Data Preprocessing**
   - Extracts features from multiple audio files.
   - Standardizes data using Min-Max scaling.
   - Saves processed datasets for reuse.

3. **Machine Learning Models**
   - Implements classification algorithms:
     - Gaussian Naive Bayes
     - Logistic Regression
     - Decision Trees
     - Random Forest
     - K-Nearest Neighbors
   - Evaluates models with accuracy and confusion matrices.
   - Uses validation curves to tune hyperparameters for Random Forest and K-Nearest Neighbors.

4. **Predictions**
   - Makes predictions on unseen data.
   - Exports prediction results to a CSV file.

---

## Setup

### Prerequisites
Make sure the following libraries are installed:
- Python 3.8+
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `librosa`
- `scikit-learn`
- `soundfile`

Install the dependencies via pip:
```bash
pip install numpy pandas matplotlib seaborn librosa scikit-learn soundfile
```

### File Structure
The project expects the following directory structure:
```
project-folder/
|-- Data/                    # Folder containing `.wav` audio files
|-- audio_features_with_year.csv  # Extracted audio features (auto-generated)
|-- audio_features_scaled.csv     # Scaled dataset (auto-generated)
|-- audios_predicciones.csv       # New audio data for predictions
|-- script.py                 # Main script (this code)
```

---

## Usage

### 1. Extract Audio Features
- Place your `.wav` audio files in the `Data/` folder.
- Run the script to extract features from the audio files.
- The extracted features are saved to `audio_features_with_year.csv`.

### 2. Preprocess Data
- The script standardizes the features and saves the scaled data to `audio_features_scaled.csv`.

### 3. Train and Evaluate Models
- The script trains five classifiers and evaluates their accuracy.
- Accuracy and confusion matrices for each model are displayed.

### 4. Make Predictions
- Place a file named `audios_predicciones.csv` in the project folder containing new audio features.
- Run the script to generate predictions. Results are saved in `predicciones_con_features.csv`.

---

## Key Functions
- **Feature Extraction:** Processes audio files and extracts features including duration, magnitudes, MFCCs, and Mel-spectrogram values.
- **Scaling:** Normalizes features between 0 and 1 for machine learning.
- **Classification:** Uses scikit-learn models to predict the year of the audio file.
- **Visualization:** Generates plots for spectrograms, correlation matrices, and validation curves.

---

## Outputs
1. **Datasets**:
   - `audio_features_with_year.csv`: Raw features extracted from the audio files.
   - `audio_features_scaled.csv`: Scaled features ready for machine learning.

2. **Plots**:
   - Spectrograms (STFT and Mel-spectrogram).
   - Correlation matrix heatmap.
   - Validation curves for hyperparameter tuning.

3. **Prediction Results**:
   - `predicciones_con_features.csv`: Predictions for unseen data.

---

## Example

### Run the Script
```bash
python script.py
```

### Output Example
```
Largo del array: 12345678
Frecuencia de muestreo: 44100
Duración: 280.5 segundos
Dataset generado con éxito. Guardado en 'audio_features_with_year.csv'.
Los datos han sido escalados y guardados en 'audio_features_scaled.csv'.
The accuracy of RandomForestClassifier is 0.9124
Resultados exportados a 'predicciones_con_features.csv'.
```

---

## Notes
- Ensure audio files have consistent naming conventions, as the script extracts the year from filenames using regex (e.g., `Canto1_2019.WAV`).
- Adjust hyperparameters (e.g., `n_estimators` for Random Forest) as needed to optimize performance.
- Use the correlation matrix to identify highly correlated features and improve feature selection.

---

## License
This project is licensed under the MIT License.

---

## Contact
For questions or feedback, contact [Your Name/Email].



## Future Work

- Extend the dataset with additional labeled audio files.
- Experiment with advanced deep learning techniques for better feature extraction and classification.


