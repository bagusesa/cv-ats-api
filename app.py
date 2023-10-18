from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.svm import SVC

app = Flask(__name__)

# Load the trained model
clf = joblib.load('cv-model.pkl')

# Load the tfidf vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    df = pd.read_excel(file)
    df = df.astype(str)

    X = df.drop(columns=['label','nama'])
    y = df[['label','nama']]

    data = vectorizer.transform(X.apply(lambda x: ' '.join(x), axis=1))

    prediction = clf.predict(data)
    pred_score = clf.predict_proba(data)

    df_score = pd.DataFrame(pred_score, columns = ['Data Scientist','Designer','Developer','Engineer','General','Marketing','Researcher'])
    df_pred = pd.DataFrame(prediction, columns = ['Recommendation Role'])

    label_map = {0: 'Data Scientist', 1: 'Designer', 2: 'Developer', 3: 'Engineer', 4: 'General', 5: 'Marketing', 6: 'Researcher'}
    df_pred['Recommendation Role'] = df_pred['Recommendation Role'].map(label_map)

    df_final = pd.concat([y, df_pred, df_score], axis=1)

    df_final.to_excel('result.xlsx', index=False)

    output = {'result': 'success'}
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
