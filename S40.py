import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify,render_template
from datetime import datetime  # Import datetime module

app = Flask(__name__)

# Load the trained model
with open('resultat.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route('/predicts400', methods=['POST'])
def predicts():
    
    data = request.get_json()
    # Convert JSON data to a 2D NumPy array
    #data_array = np.array(list(data.values())).reshape(1, -1)

     # Prepare input data for prediction
    input_data = pd.DataFrame(data, index=[0])
    input_data['Date_comptable'] = pd.to_datetime(input_data['Date_comptable'])
    input_data['Année'] = input_data['Date_comptable'].dt.year
    input_data['Mois'] = input_data['Date_comptable'].dt.month
    input_data['Jour'] = input_data['Date_comptable'].dt.day
    input_data = input_data.drop('Date_comptable', axis=1)

    # Make prediction
    prediction = loaded_model.predict(input_data)


    # Return the prediction as JSON
    return jsonify({'prediction': prediction.tolist()})


@app.route('/predicts40', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        code_article = int(request.form['Code_Article'])
        code_site = int(request.form['Code_Site'])
        date_comptable = request.form['Date_comptable']
        Code_Fam_Stat_Article_1 = int(request.form['Code_Fam_Stat_Article_1'])
        Code_Fam_Stat_Article_2 = int(request.form['Code_Fam_Stat_Article_2'])
        Code_Fam_Stat_Article_3 = int(request.form['Code_Fam_Stat_Article_3'])
        Affaire = int(request.form['Affaire'])
        Mt_Ligne_HT = float(request.form['Mt_Ligne_HT'])

        # Prepare input data for prediction
        input_data = pd.DataFrame({
            'Code_Article': [code_article],
            'Code_Site': [code_site],
            'Date_comptable': [date_comptable],
            'Code_Fam_Stat_Article_1': [Code_Fam_Stat_Article_1],
            'Code_Fam_Stat_Article_2': [Code_Fam_Stat_Article_2],
            'Code_Fam_Stat_Article_3': [Code_Fam_Stat_Article_3],
            'Affaire': [Affaire],
            'Mt_Ligne_HT': [Mt_Ligne_HT]
        })

        # Convert Date_comptable to datetime and add year, month, day columns
        input_data['Date_comptable'] = pd.to_datetime(input_data['Date_comptable'])
        input_data['Année'] = input_data['Date_comptable'].dt.year
        input_data['Mois'] = input_data['Date_comptable'].dt.month
        input_data['Jour'] = input_data['Date_comptable'].dt.day
        input_data = input_data.drop('Date_comptable', axis=1)

        # Make prediction
        prediction = loaded_model.predict(input_data)

        return render_template('prediction_result.html', prediction=prediction[0])
    else:
        return render_template('predict_form.html')


if __name__ == '__main__':
    app.run(debug=True)
