from flask import Flask, render_template, request
from joblib import load
import numpy as np
import pandas as pd

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def make_prediction():
    feature = ['gender','age', 'hypertension', 'heart_disease', 'ever_married', 'work_type','Residence_type','avg_glucose_level','bmi', 'smoking_status']

    model = load('model.joblib')
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', prediction = "")
    else:
        Gender = int(request.form['Gender'])
        Age = float(request.form['Age'])
        Hypertension = int(request.form['hypertension'])
        HeartDistance = int(request.form['heart_disease'])
        EverMarried = int(request.form['ever_married'])
        WorkType = int(request.form['work_type'])
        ResidenceType = int(request.form['Residence_type'])
        AvgGlucoseLevel = float(request.form['avg_glucose_level'])
        Bmi = float(request.form['bmi'])
        Smoking = int(request.form['smoking_status'])

        test_np_input = np.array([[Gender, Age, Hypertension, HeartDistance, 
                                   EverMarried, WorkType, ResidenceType, 
                                   AvgGlucoseLevel, Bmi, Smoking]])
        input_df = pd.DataFrame(test_np_input, columns=feature)

        pred = model.predict(input_df).tolist()
        pred_as_str = str(pred)
        if(pred_as_str == '[0]'):
            predictions = 'Chúc mừng bạn không có nguy cơ bị đột quỵ ^-^'
        else:
            predictions = 'Hãy bảo vệ sức khỏe bạn đang có nguy cơ bị đột quỵ !!!'
        return render_template('index.html', prediction = predictions)

if __name__ == "__main__":
    app.run(debug=True)
