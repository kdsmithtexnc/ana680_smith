from flask import Flask, render_template, request
import numpy as np
import pickle
import joblib
app = Flask(__name__)
filename = 'file_cancer.pkl'
#model = pickle.load(open(filename, 'rb'))
svm_rbf = joblib.load(filename)
#model = joblib.load(filename)
@app.route('/')
def index(): 
    return render_template('index.html')
@app.route('/predict', methods=['POST'])
def predict():
    Uniformity_of_Cell_Size = request.form['Uniformity_of_Cell_Size']
    Uniformity_of_Cell_Shape = request.form['Uniformity_of_Cell_Shape']
    Bare_Nuclei = request.form['Bare_Nuclei']
    
      
    pred = svm_rbf.predict(np.array([[Uniformity_of_Cell_Size, Uniformity_of_Cell_Shape, Bare_Nuclei ]]))
    print(pred)
    return render_template('index.html', predict = str(pred))


if __name__ == '__main__':
    app.run
