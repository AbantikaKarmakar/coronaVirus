from flask import Flask, render_template, request
from flask.wrappers import Request
app = Flask(__name__)
import pickle

# open a file, where you stored the pickled data
file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])
def hello_world():
    if request.method == "POST":
        myDict = request.form
        Fever = int(myDict['Fever'])
        Age = int(myDict['Age'])
        BodyPain = int(myDict['BodyPain'])
        RunnyNose = int(myDict['RunnyNose'])
        DiffBreath = int(myDict['DiffBreath'])
        # Code for inference
        inputFeatures = [Fever, BodyPain, Age, RunnyNose, DiffBreath]
        infProb = clf.predict_proba([inputFeatures])[0][1]
        print(infProb)
        return render_template('show.html', inf=round(infProb*100))
    return render_template('index.html')
    # return 'Hello, World!' + str(infProb)


if __name__ == "__main__":
    app.run(debug=True)