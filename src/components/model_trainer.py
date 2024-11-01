import os
import sys
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object

import pandas as pd

#modeling
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
#metrics 
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score


#validation
from sklearn.model_selection import GridSearchCV

@dataclass 
class ModelTrainerConfig():
    model_path = os.path.join("artifacts" , "model.pkl")

class ModelTrainer():
    def __init__(self):
        self.dataconfig = ModelTrainerConfig()
        self.models = {
                     "LogisticRegression": LogisticRegression(),
                    "AdaBoostClassifier": AdaBoostClassifier(),
                    "XGBClassifier": XGBClassifier(),
                    "KNeighborsClassifier": KNeighborsClassifier()
                }
        
        self.models_params = {
                        "LogisticRegression": {
                        'C': [0.1, 1],
                        'solver': ['liblinear', 'saga'],  # Agrega solver para compatibilidad
                        'penalty': ['l1', 'l2'],
                        'max_iter': [200, 500]
                        },
                        "AdaBoostClassifier": {
                            'n_estimators': [50, 100],
                            'learning_rate': [0.1, 0.5]  ,    
                            'algorithm': ['SAMME']  
                        },
                        "XGBClassifier": {
                            'n_estimators': [50, 100],
                            'learning_rate': [0.1, 0.2],
                            'max_depth': [5, 7]
                        },
                        "KNeighborsClassifier": {
                            'n_neighbors': [3, 5, 7 , 9 , 12],       # Número de vecinos
                            'weights': ['uniform', 'distance'],  # Peso de los puntos
                            'metric': ['euclidean', ] # Métrica de distancia
                        }
                                    }
        self.report =  None
        self.threshold = 0.6
        
    def show_report(self):
        print(self.report)
    
    def evaluate_model(self ,X_train , y_train , X_test , y_test):
        try:
            report = []
            for model_name , model in self.models.items():
                params = self.models_params[model_name]
                gs = GridSearchCV(model ,params  , cv = 3)
                gs.fit(X_train , y_train)

                model.set_params(**gs.best_params_)

                model.fit(X_train , y_train)

                y_test_pred = model.predict(X_test)

                accuracy  = accuracy_score(y_test ,y_test_pred)
                recall = recall_score(y_test , y_test_pred , average = "macro")
                precision = precision_score(y_test , y_test_pred , average = "macro")
                f1 = f1_score(y_test , y_test_pred , average="macro")
                average_score = ( accuracy+recall +  precision+  f1) / 4
                
                report.append({
                    "model_name":model_name , 
                    "accuracy":accuracy , 
                    "recall":recall , 
                    "precision":precision , 
                    "f1_score":f1, 
                    "average_score" : average_score 
                })
            
            report = pd.DataFrame(report)
            report = report.sort_values(by ="accuracy" , ascending=False)
            self.report = report
            return report
        except Exception as e:
            raise CustomException(e , sys)
    
    def initate_model_training(self ,train_array , test_array ):
        
        try:
            X_train , y_train , X_test , y_test = (
                train_array[:, : -1] ,  train_array[:, -1] ,test_array[: , :-1]  ,test_array[: , -1]  
                            )
            
            model_report_df:pd.DataFrame = self.evaluate_model(X_train , y_train , X_test , y_test)
            
            best_model_row = model_report_df.loc[model_report_df["accuracy"].idxmax()]
            best_model_name = best_model_row["model_name"]
            best_model_score = best_model_row["accuracy"]
            best_model = self.models[best_model_name]

            if best_model_score<=self.threshold:
                raise ValueError("best model couldn't be found")
            
            logging.info("best model found")
            save_object(file_path = self.dataconfig.model_path , obj=best_model)
            
            return (best_model_name , best_model_score , best_model )
        except Exception as e:  
            raise CustomException(e , sys)