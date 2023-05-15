import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            model_path=os.path.join('artifacts','model.pkl')

            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)

            data_scaled=preprocessor.transform(features)

            pred=model.predict(data_scaled)
            return pred
            

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,
                 age:float,
                 education_num:float,
                 capital_gain:float,
                 hours_per_week:float,
                 workclass:str,
                 education:str,
                 marital_status:str,
                 occupation:str,
                 relationship:str,
                 race:str,
                 sex:str,
                 native_country:str,
                 fnlwgt:float,
                 capital_loss:float):

        
        self.age=age
        self.education_num=education_num
        self.capital_gain=capital_gain
        self.hours_per_week=hours_per_week
        self.workclass=workclass
        self.education = education
        self.marital_status = marital_status
        self.occupation = occupation
        self.relationship = relationship
        self.race = race
        self.sex = sex
        self.native_country = native_country
        self.fnlwgt = fnlwgt
        self.capital_loss = capital_loss

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                'age':[self.age],
                'education_num':[self.education_num],
                'capital_gain':[self.capital_gain],
                'hours_per_week':[self.hours_per_week],
                'workclass':[self.workclass],
                'education':[self.education],
                'marital_status':[self.marital_status],
                'occupation':[self.occupation],
                'relationship':[self.relationship],
                'race':[self.race],
                'sex':[self.sex],
                'native_country':[self.native_country],
                'fnlwgt':[self.fnlwgt],
                'capital_loss':[self.capital_loss]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df
        
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)

