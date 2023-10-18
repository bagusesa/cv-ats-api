# Machine Learning Model Deployment with Flask

This project demonstrates a simple deployment of a machine learning model using Flask. The application takes an Excel file as input, applies a trained SVM-based classification model to make predictions on the data, and then saves the results to an Excel file. The model predicts the recommendation role based on the input data.

## Requirements

- Python 3.8
- Flask
- pandas
- scikit-learn 1.2.2
- joblib

## Installation

1. Clone the repository:

   ```bash
   git clone <repository_url>
   ```

2. Navigate to the project directory:

   ```bash
   cd <project_directory>
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Place the trained model file, `cv-model.pkl`, and the TF-IDF vectorizer file, `tfidf_vectorizer.pkl`, in the project directory.

2. Run the Flask application:

   ```bash
   python app.py
   ```

3. The application will run on `http://localhost:5000/`.

4. To make a prediction, send a POST request to `http://localhost:5000/predict` with an Excel file containing the required data.

5. The application will generate an Excel file named `result.xlsx` with the predicted results.
