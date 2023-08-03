import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from flask import Flask, request, jsonify
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

app = Flask(__name__)

def train_model():
    d={1:'IT IS A FRAUD TRANSACTION',0:'IT IS NOT A FRAUD TRANSACTION'}
    # Load your dataset
    data = pd.read_csv("creditcard.csv")
    print(data.info())
    y=data['Class'].apply(lambda x:1 if x=='1' else 0)
    x=data.drop('Class',axis=1)


    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25)
    #
    # x_train=sc.fit_transform(x_train)
    # x_test=sc.fit_transform(x_test)
    # help(RandomForestClassifier )

    rf=RandomForestClassifier()

    rf.fit(x_train,y_train)
    p=rf.predict(x_test)
    v=rf.predict([[10,1.44904378114715,-1.17633882535966,0.913859832832795,-1.37566665499943,-1.97138316545323,-0.62915213889734,-1.4232356010359,0.0484558879088564,-1.72040839292037,1.62665905834133,1.1996439495421,-0.671439778462005,-0.513947152539479,-0.0950450453999549,0.230930409124119,0.0319674667862076,0.253414715863197,0.854343814324194,-0.221365413645481,-0.387226474431156,-0.00930189652490052,0.313894410791098,0.0277401580170247,0.500512287104917,0.25136735874921,-0.129477953726618,0.0428498709381461,0.0162532619375515,7.8]])
    #pickle.dump(rf,open('model1.pkl','wb'))
    print(d[v[0]])
    print(rf.score(x_train,y_train)*100,"%")
    print(accuracy_score(y_test,p)*100,'%')
    # print(confusion_matrix(y_test,predict))

  

    # Save the trained model using joblib
    joblib.dump(rf, 'model.joblib')

# Check if the model file exists; if not, train and save the model
try:
    model = joblib.load('model.joblib')
except FileNotFoundError:
    print("Model not found. Training the model...")
    train_model()
    model = joblib.load('model.joblib')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the input data from the JSON request
        data = request.get_json()

        # Perform any necessary preprocessing on the input data
        # (Ensure that the data is in the same format as during training)

        # Make the prediction using your model
        prediction = model.predict(data)

        # Return the prediction as JSON
        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
