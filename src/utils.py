import os
import sys
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise CustomException(e,sys)
    

def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]

            # Train model
            model.fit(X_train, y_train)

            # Predict train and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Get accuracy scores for train and test data
            train_model_score = accuracy_score(y_train, y_train_pred)
            test_model_score = accuracy_score(y_test, y_test_pred)

            # Get classification report and confusion matrix fot test data
            classificationReport = classification_report(y_test,y_test_pred,output_dict=True)
            confusionMatrix = confusion_matrix(y_test,y_test_pred)

            report[list(models.keys())[i] ] = dict({"Accuracy_score_of_train_data":train_model_score,"Accuracy_score_of_test_data":test_model_score, "ClassificationReport": classificationReport, "ConfusionMatrix":confusionMatrix})

        return report
    
    except Exception as e:
        logging.info("Exception occured during model training")
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    
    except Exception as e:
        logging.info("Exception Occured in load_object function utils")
        raise CustomException(e,sys)
    

def iterdict(dictionary):
    for key,val in dictionary.items():        
        if isinstance(val, dict):
           iterdict(val)
        else:            
            print (key,":",val)