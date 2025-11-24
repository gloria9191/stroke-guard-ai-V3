
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('stroke_model.pkl','rb'))

FEATS = ["Sex","Age","BMI","SBP","DBP","Glucose","Smoking","Alcohol"]

@app.route('/', methods=['GET','POST'])
def index():
    pred=None
    if request.method=='POST':
        vals=[]
        for f in FEATS:
            vals.append(float(request.form[f]))
        prob = model.predict_proba(np.array([vals]))[0][1]
        pred = round(float(prob),4)
    return render_template('index.html', pred=pred)

if __name__=='__main__':
    app.run(host='0.0.0.0',port=8000)
