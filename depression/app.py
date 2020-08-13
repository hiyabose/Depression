from flask import Flask, request, url_for, redirect, render_template
import pickle
import numpy as np

app = Flask(__name__, template_folder="template")

reg = pickle.load(open("depression/model.pkl", "rb"))


@app.route("/")
def hello_world():
    return render_template("hel.html")


@app.route("/predict", methods=["POST"])
def home():
    data1 = float(request.form["a"])
    data2 = float(request.form["b"])
    data3 = float(request.form["c"])
    data4 = float(request.form["d"])
    d5 = float(request.form["e"])
    d6 = float(request.form["f"])
    d7 = float(request.form["g"])
    d8 = float(request.form["h"])
    d9 = float(request.form["i"])
    d10 = float(request.form["j"])
    d11 = float(request.form["k"])
    d12 = float(request.form["l"])
    d13 = float(request.form["m"])
    d14 = float(request.form["n"])
    d15 = float(request.form["o"])
    d16 = float(request.form["p"])
    d17 = float(request.form["q"])
    d18 = float(request.form["r"])
    d19 = float(request.form["s"])
    d20 = float(request.form["t"])
    d21 = float(request.form["u"])

    arr = np.array(
        [
            [
                data1,
                data2,
                data3,
                data4,
                d5,
                d6,
                d7,
                d8,
                d9,
                d10,
                d11,
                d12,
                d13,
                d14,
                d15,
                d16,
                d17,
                d18,
                d19,
                d20,
                d21,
            ]
        ]
    )
    pred = reg.predict(arr)
    print(pred)
    return render_template("after.html", data=pred)


if __name__ == "__main__":
    app.run(debug=True)
