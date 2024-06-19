# After, you can give app.py file to DevOps for deployment of the model.

import pickle
import pandas as pd
import numpy as np
from datetime import datetime
import sklearn

class Inferance:
    def __init__(self,model_path,scaler_path):

        self.model = pickle.load(open(model_path,'rb'))
        self.sc = pickle.load(open(scaler_path, 'rb'))

    def prediction(self,scaled_data):
        pred_result = self.model.predict(scaled_data)
        # print(pred_result)
        return pred_result
    def scalling(self,df):
        scaled_data = self.sc.transform(df)
        # print(scaled_data)
        return self.prediction(scaled_data)

    def user_input_to_model_input(self,user_input):

      # input :
      # {'date':'10/06/2023','hour':21,'temperature':25',humidity':55,'wind_speed':67,'visibility':25,
      # 'solar_radiation':0.0,'rainfall':0.7,'snowfall':0.2,'seasons':"Winter",'holiday':"No Holiday",'functioning_day':"Yes"}
      input = user_input
      #-----------------------------------------------------------------------------------------------------------------------
      # first convert holiday and functioning_day
      holiday_opt = {'No Holiday':0,'Holiday':1}
      functioning_day_opt = {'Yes':1,'No':0}

      holiday_choice = holiday_opt[input[10]]
      functioning_day_choice = functioning_day_opt[input[11]]

      #-----------------------------------------------------------------------------------------------------------------------
      # Now encoding the date
      dt = datetime.strptime(input[0],"%d/%m/%Y")
      day  = dt.day
      month = dt.month
      year = dt.year
      day_name = dt.strftime('%A')

      #-----------------------------------------------------------------------------------------------------------------------
      # Dataframe :1 ---> other_df
      # Feature sequence to be pass to model
      feature_squence_list = ['Hour', 'Temperature(°C)', 'Humidity(%)', 'Wind speed (m/s)','Visibility (10m)',
                              'Solar Radiation (MJ/m2)', 'Rainfall(mm)','Snowfall (cm)', 'Holiday',
                              'Functioning Day', 'Day', 'Month', 'Year']

      # user values to be passed in the sequence of above.
      user_input =input[1:9]
      other_input =[holiday_choice,functioning_day_choice,day,month,year]
      user_input.extend(other_input)

      other_df = pd.DataFrame(columns=feature_squence_list,data=[user_input])

      #-----------------------------------------------------------------------------------------------------------------------
      # Dataframe:2-----> seasons_df
      # Now handle the Seasons
      seasons_col = ['Spring', 'Summer', 'Winter']
      saeasons_data = np.zeros((1,len(seasons_col)),dtype=int)

      seasons_df = pd.DataFrame(columns=seasons_col,data=saeasons_data)

      if input[9] in seasons_col:    # input[9] --> 'Winter'
        seasons_df[input[9]] = 1

      #-----------------------------------------------------------------------------------------------------------------------
      # Dataframe: 3 ---> week_day_df
      # Now handle the weekday..
      week_day_col = ['Monday', 'Saturday', 'Sunday','Thursday', 'Tuesday', 'Wednesday']
      week_day_data = np.zeros((1,len(week_day_col)),dtype=int)

      week_day_df = pd.DataFrame(columns=week_day_col,data=week_day_data)

      if day_name in week_day_col:
        week_day_df[day_name] = 1

      #-----------------------------------------------------------------------------------------------------------------------
      # Concate the 3 dataFrame and make it 1 DataFrame for model input
      test_df = pd.concat([other_df,seasons_df,week_day_df],axis=1)
      # print(test_df)

      #-----------------------------------------------------------------------------------------------------------------------
      # scalling the data
      #scale=
      return self.scalling(test_df)

    def user_input(self):
        print('Enter the coorect Information to Predict the Bike Deman over Date and Time.')
        # input
        date = input('Date (dd/mm/yyyy):')
        hour = int(input('Hour (0-23):'))
        temperature = float(input('Temperature in C:'))
        humidity = float(input('Humidity :'))
        wind_speed = float(input('Wind Speed :'))
        visibility = float(input('Visibility :'))
        solar_radiation = float(input('Solar_radiation :'))
        rainfall = float(input('Rainfall :'))
        snowfall = float(input('Snowfall :'))
        seasons = input('Seasons (Antum/Spring/Summer/Winter):')
        holiday = input('Holiday (No Holiday/Holiday):')
        functioning_day = input('Functioning_day (Yes/No) :')

        data = [date,hour,temperature,humidity,wind_speed,visibility,solar_radiation,
                rainfall,snowfall,seasons,holiday,functioning_day]

        # result_data =
        return self.user_input_to_model_input(data),date,hour

if __name__ == "__main__":
    inf = Inferance('xgboost_regressor_r3_0_93_v1.pkl','StandardScaler.pkl')
    result,date,hour = inf.user_input()
    if result[0] <= 0:
        print(f"Rented Bike Demand on Date : {date} & Time : {hour} is : 0 ")
    else:
        print(f"Rented Bike Demand on Date : {date} & Time : {hour} is : {int(result[0].round())}")