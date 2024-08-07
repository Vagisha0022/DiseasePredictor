from flask import Flask, render_template, redirect, url_for,request,session
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import pandas as pd
import pickle
import numpy as np
app = Flask(__name__)
db = SQLAlchemy()
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db.init_app(app)
app.secret_key= 'secret_key'
class user(db.Model):
    id= db.Column(db.Integer,primary_key=True)
    name= db.Column(db.String(100),nullable=False)
    email= db.Column(db.String(100),unique=True)
    password= db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name= name
        self.email = email
        self.password= bcrypt.hashpw(password.encode('utf-8'),bcrypt.gensalt()).decode('utf-8')
    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))
with app.app_context(): 
    db.create_all() 


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/help')
def help():
    return render_template('help.html')

@app.route('/terms')
def terms():
    return render_template("tc.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = user(name=name, email=email, password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect(url_for('login'))  # Redirect to login page after successful registration

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        logged_user = user.query.filter_by(email=email).first()
        if logged_user and logged_user.check_password(password):
            session['name'] = logged_user.name
            session['email'] = logged_user.email
            session['password'] = logged_user.password
            return redirect('/dashboard')
        else:
            return render_template('login.html', error='Invalid user')
    return render_template('login.html')



@app.route('/dashboard')
def dashboard():
    if 'name' in session:
        logged_user = user.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html', logged_user=logged_user)
    else:
        return redirect(url_for('login'))
    
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
@app.route("/liver")
def liver():
    return render_template("liver.html")
@app.route('/predictliver', methods=['POST'])
def predictliver():
    input_features = [int(x) for x in request.form.values()]
    features_value = [np.array(input_features)]
    features_name = ['Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase',
                     'Alamine_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio']
    df = pd.DataFrame(features_value, columns=features_name)
    modelLIVER = pickle.load(open('modelLIVER.pkl', 'rb'))
    output = modelLIVER.predict(df)
    if output == 1:
        res_val = "a high risk of Liver Disease"
    else:
        res_val = "a low risk of Liver Disease"

    return render_template('liver_result.html', prediction_text='Patient has {}'.format(res_val))
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
@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/login')
if __name__ == '__main__':
    app.run(debug=True)