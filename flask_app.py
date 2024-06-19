import pandas as pd
import numpy as np
from flask import Flask,request,render_template
from app import Inferance

app = Flask(__name__)

result_df = pd.DataFrame(columns = ['Date','Hour','Temperature (°C)','Humidity (%)','Wind Speed (km/h)',
                                   'Visibility (km)','Solar Radiation (MJ/m²)','Rainfall (mm)',
                                    'Snowfall (cm)','Seasons','Holiday','Functioning Day',
                                    'Rented Bike Count'])

@app.route("/")
def home():
    return render_template('index.html')

@app.route("/predict", methods=['post'])
def predict():
    global result_df

    data = [x for x in request.form.values()]
    # data = ['10/06/2023', '21', '25', '55', '67', '25', '0', '0.7',
    #         '0.2', 'Winter', 'No Holiday', 'Yes']

    input = []
    for i, j in enumerate(data):
        if i in range(1, 9):
            input.append(float(j))
        else:
            input.append(j)

    inf = Inferance('xgboost_regressor_r3_0_93_v1.pkl','StandardScaler.pkl')
    result = inf.user_input_to_model_input(input)
    f_result = int(result[0].round())
    input.append(f_result)

    result_df.loc[len(result_df)] = input
    print(result_df)

    result_df.to_csv('bike_demand_predicted_data.csv',index=False,header=False,mode='a')



    if f_result <= 0:
        return render_template('index.html',
                               prediction_text=f"Rented Bike Demand on Date : {input[0]} & Time : {input[1]} is : 0 ")
    else:
        return render_template('index.html',
                               prediction_text=f"Rented Bike Demand on Date : {input[0]} & Time : {input[1]} is : {f_result} ")


if __name__ == '__main__':
    app.run(port=5000)# Only for aws....host='0.0.0.0',port = 8080


