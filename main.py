from flask import Flask, render_template, request
from flask_bootstrap import Bootstrap
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)
Bootstrap(app)

@app.route("/",methods=["POST","GET"])
def home():
    data = pd.read_csv('data.csv')

    X = data.drop(columns=['Desease'])
    y = data['Desease']

    model = DecisionTreeClassifier()

    model.fit(X.values, y)
    
    # fever = request.form["fever"]
    # cough = request.form["cough"]
    # fatigue = request.form["fatigue"]
    # difBr = request.form["difBr"]
    # age = request.form["age"]
    # gender = request.form["gender"]
    # bp = request.form["bp"]
    # chol = request.form["chol"]

    diagnosis = model.predict([[1, 1, 0, 1, 20, 1, 3, 1]])

    return render_template("home.html", diagnosis=diagnosis)

if __name__ == "__main__":
    app.run(debug=True)