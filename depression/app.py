
from flask import Flask,request, url_for, redirect, render_template
import pickle
import numpy as np

app=Flask(__name__,template_folder='template')

reg=pickle.load(open('model.pkl','rb'))

@app.route('/')
def hello_world():
    return render_template('hel.html')

@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['a']
    data2 = request.form['b']
    data3 = request.form['c']
    data4 = request.form['d']
    d5 = request.form['e']
    d6 = request.form['f']
    d7 = request.form['g']
    d8 = request.form['h']
    d9= request.form['i']
    d10= request.form['j']
    d11 = request.form['k']
    d12 = request.form['l']
    d13 = request.form['m']
    d14 = request.form['n']
    d15 = request.form['o']
    d16 = request.form['p']
    d17 = request.form['q']
    d18 = request.form['r']
    d19 = request.form['s']
    d20 = request.form['t']
    d21 = request.form['u']

    arr = np.array([[data1, data2, data3, data4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,d21]])
    pred = reg.predict(arr)
    return render_template('after.html', data=pred)


if __name__=="__main__":
    app.run(debug=True)