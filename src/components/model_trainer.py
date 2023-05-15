import sys
import os
# import pprint
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model


@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Splitting Dependent and Independent variables from train and test data")
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],             # all independent array 
                train_array[:,-1],              # dependent array
                test_array[:,:-1],              # all independent array 
                test_array[:,-1],               # dependent array

            )

            models = {
                'LogisticRegression': LogisticRegression(),
                'DecisionTreeClassifier': DecisionTreeClassifier(),
                'GaussianNB': GaussianNB(),
                'KNeighborsClassifier': KNeighborsClassifier(),
                'SVC': SVC()
            }

            ## evaluting model report
            model_report:dict = evaluate_model(X_train, y_train, X_test, y_test,models)
            # print(model_report)
            for key,_ in model_report.items():
                print()
                print(key)
                print("-"*20)
                print(f"Accuracy score of test data: {model_report[key]['Accuracy_score_of_test_data']}")
                print(f"Classification Report Model Accuracy : {model_report[key]['ClassificationReport']['accuracy']}")
                print(f"Confusion Matrix : {model_report[key]['ConfusionMatrix']}")
            # pp = pprint.PrettyPrinter(depth=5)
            # pp.pprint(model_report)
                

            print("\n=====================================================================================================\n")
            logging.info(f"Model Report : {model_report}")

            ## To get best model score and name from dictionary
            # max_accuracy_key = max(model_report, key=lambda x: (model_report[x]["ClassificationReport"]["accuracy"]))
            #            
            best_model_name = max(model_report, key=lambda x: (model_report[x]["ClassificationReport"]["accuracy"]))
            best_model_score = model_report[best_model_name]["ClassificationReport"]["accuracy"]
            
            best_model = models[best_model_name]

            print(f"Best Model Found ! \n  Model Name : {best_model_name} , Accuracy : {best_model_score}")
            print("\n=====================================================================================================\n")
            logging.info(f"Best Model Found ! \n Model Name : {best_model_name} , Accuracy : {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

        except Exception as e:
            logging.info("Exception occured at Model Training")
            raise CustomException(e,sys)