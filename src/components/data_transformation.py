import sys
import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer                                # Handling Missing Values
from sklearn.preprocessing import StandardScaler, OneHotEncoder      # Handling Feature Scaling & Ordinal Encoding 
from sklearn.pipeline import Pipeline                                   # Pipelining
from sklearn.compose import ColumnTransformer                           # Column Transformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

import warnings
warnings.filterwarnings("ignore")

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object(self):
        try:
            logging.info('Data Transformation Starts')

            # Define columns which should be encoded and which should be scaled
            categorical_cols = ['workclass', 'education', 'marital_status', 'occupation','relationship', 'race', 'sex', 'native_country']
            numerical_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 'capital_loss','hours_per_week']

            logging.info("Pipeline Initiated")

            ## Numerical Pipeline
            num_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
                ]
            )
            
            ## Categorical Pipeline
            cat_pipeline = Pipeline(
                steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinalencoder', OneHotEncoder(sparse_output=False)),
                # ('scaler', StandardScaler())
                ]
            )

            ## Column transforming
            preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols)
            ])
            
            logging.info("Pipeline Completed")
            return preprocessor
        
            # logging.info("Pipeline Completed")

        except Exception as e:
            logging.info("Error in Data Transformation")
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            ## Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed ")
            logging.info(f"Train Dataframe Head : \n{train_df.head().to_string()}")
            logging.info(f"Test Dataframe Head : \n{test_df.head().to_string()}")

            logging.info("Obtaining Preprocessing object ")

            preprocessing_obj = self.get_data_transformation_object()

            target__column_name = 'income'
            drop_columns = [target__column_name]
            
            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target__column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target__column_name]
                                             
            ## Transformating using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying Preprocessing object on training and testing Datasets.")

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # saving pickle file
            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info("Preprocessor pickle file saved")

            # returning info for model training
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")

            raise CustomException(e,sys)

