from flask import Flask, request, jsonify
import joblib
import pandas as pd
import json

app = Flask(__name__)

# Load the trained model
clf = joblib.load('cv-model.pkl')

# Load the tfidf vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = json.loads(request.data)

    rows = []
    for obj in data:
        # Extract the relevant data from the object
        name = obj['participant']['name']
        title1 = obj['internshipRoles'][0]['title']
        title2 = obj['internshipRoles'][1]['title'] if len(
            obj['internshipRoles']) > 1 else None
        title3 = obj['internshipRoles'][2]['title'] if len(
            obj['internshipRoles']) > 2 else None
        educational_backgrounds = obj['participant']['curriculumVitae']['educationalBackgrounds']
        organizational_experience = obj['participant']['curriculumVitae']['organizationExperiences']
        work_experiences = obj['participant']['curriculumVitae']['workExperiences']
        skill = obj['participant']['curriculumVitae']['skill']
        final_task = obj['participant']['curriculumVitae']['finalTask']
        achievement = obj['participant']['curriculumVitae']['achievements']

        # Create a new row with the extracted data
        row = {
            'Name': name,
            'Role 1': title1,
            'Role 2': title2,
            'Role 3': title3,
            'Riwayat Pendidikan': educational_backgrounds,
            'Pengalaman Organisasi': organizational_experience,
            'Pengalaman Bekerja': work_experiences,
            'Skill': skill,
            'Tugas Akhir': final_task,
            'Prestasi': achievement
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.astype(str)

    X = df.drop(columns=['role','nama'])
    y = df[['role','nama']]

    data = vectorizer.transform(X.apply(lambda x: ' '.join(x), axis=1))

    prediction = clf.predict(data)
    pred_score = clf.predict_proba(data)

    df_score = pd.DataFrame(pred_score, columns = ['Data Scientist','Designer','Developer','Engineer','General','Marketing','Researcher'])
    df_pred = pd.DataFrame(prediction, columns = ['Recommendation Role'])

    label_map = {0: 'Data Scientist', 1: 'Designer', 2: 'Developer', 3: 'Engineer', 4: 'General', 5: 'Marketing', 6: 'Researcher'}
    df_pred['Recommendation Role'] = df_pred['Recommendation Role'].map(label_map)

    df_final = pd.concat([y, df_pred, df_score], axis=1)

    results = df_final.to_dict(orient='records')

    output = {'result': 'success', 'data': results}
    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=False)