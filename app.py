from flask import Flask,render_template,request
import pandas as pd
import pickle
import numpy as np


app = Flask(__name__)
@app.route("/")
def dashboard():
    return render_template("dashboard.html")
@app.route("/disindex")
def disindex():
    return render_template("disindex.html")
@app.route("/cancer")
def cancer():
    return render_template("cancer.html")

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape', 'marginal_adhesion',
                     'single_epithelial_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses']
    df = pd.DataFrame(features_value, columns=features_name)
    model = pickle.load(open('model.pkl', 'rb'))
    output = model.predict(df)
    if output == 4:
        res_val = "a high risk of Breast Cancer"
    else:
        res_val = "a low risk of Breast Cancer"

    return render_template('cancer_result.html', prediction_text='Patient has {}'.format(res_val))
# diabetes
@app.route("/diabetes")
def diabetes():
     return render_template("diabetes.html")
@app.route('/predictdiabetes', methods=['POST'])
def predictdiabetes():
     input_features = [int(x) for x in request.form.values()]
     features_value = [np.array(input_features)]
     features_name = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DPF','Age']
     df = pd.DataFrame(features_value, columns=features_name)
     modelDIABETES = pickle.load(open('modelDIABETES.pkl', 'rb'))
     output = modelDIABETES.predict(df)
     if output == 1:
         res_val = "a high risk of Diabetes"
     else:
         res_val = "a low risk of Diabetes"

     return render_template('diabetes_result.html', prediction_text='Patient has {}'.format(res_val))

# liver
# @app.route("/liver")
# def liver():
#     return render_template("liver.html")
# @app.route('/predictliver', methods=['POST'])
# def predictliver():
#     input_features = [int(x) for x in request.form.values()]
#     features_value = [np.array(input_features)]
#     features_name = ['Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase',
#                      'Alamine_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']
#     df = pd.DataFrame(features_value, columns=features_name)
#     modelLIVER = pickle.load(open('modelLIVER.pkl', 'rb'))
#     output = modelLIVER.predict(df)
#     if output == 1:
#         res_val = "a high risk of Liver Disease"
#     else:
#         res_val = "a low risk of Liver Disease"

#     return render_template('liver_result.html', prediction_text='Patient has {}'.format(res_val))
@app.route("/heart")
def heart():
     return render_template("heart.html")
@app.route('/predictheart', methods=['POST'])
def predictheart():
     input_features = [int(x) for x in request.form.values()]
     features_value = [np.array(input_features)]
     features_name = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']
     df = pd.DataFrame(features_value, columns=features_name)
     modelHEART = pickle.load(open('modelHEART.pkl', 'rb'))
     output = modelHEART.predict(df)
     if output == 1:
         res_val = "a high risk of Heart Failure"
     else:
         res_val = "a low risk of Heart Failure"

     return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))

if __name__ == "__main__":
    app.run(debug=True)

