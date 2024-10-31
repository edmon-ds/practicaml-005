import os 
import sys

from src.exception import CustomException
from src.logger import logging 
from src.utils import save_object

from dataclasses import dataclass

from sklearn.base import BaseEstimator , TransformerMixin

import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from sklearn.impute import SimpleImputer

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler

import numpy as np


@dataclass
class DataTransformationConfig():
    input_preprocessor_path = os.path.join("artifacts", "input_preprocessor.pkl")
    output_encoder_path = os.path.join("artifacts", "output_encoder.pkl")

#clase para recuperar los nombres
class RestoreNameTransformer(BaseEstimator , TransformerMixin):
    def __init__(self , features_names):
        self.features_names = features_names
    def fit(self , X ):
        return self
    def transform(self, X):
        return pd.DataFrame(X , columns= self.features_names)

#clase para hacer feature engineering 
class FeatureEngineeringTransformer(BaseEstimator , TransformerMixin):        
    def __init__(self):
        pass
    def fit(self , X , y = None ):
        pass
    def transform(self , X):
        '''se crean las feature nuevas'''
        X["extra_activities_studies"] = X["Tutoring"] + X["Extracurricular"]
        X["extra_activities_no_studies"] = X["Sports"] + X["Music"] + X["Volunteering"]
        X = X.drop(columns = ["Tutoring" , "Extracurricular" , "Sports" , "Music" , "Volunteering"]) 
        return X
    def fit_transform(self, X, y = None):
        '''se crean las feature nuevas'''
        X["extra_activities_studies"] = X["Tutoring"] + X["Extracurricular"]
        X["extra_activities_no_studies"] = X["Sports"] + X["Music"] + X["Volunteering"]
        X = X.drop(columns = ["Tutoring" , "Extracurricular" , "Sports" , "Music" , "Volunteering"]) 
        return X


class DataTransformation():
    def __init__(self  ):
        self.dataconfig = DataTransformationConfig()
    def get_preprocessor(self):

        numerical_columns = ['Age', 'StudyTimeWeekly', 'Absences']

        nominal_columns = [ "Gender" ,"Ethnicity" ]

        ordinal_columns = [ 'ParentalEducation',  'ParentalSupport', 
                           "Tutoring" , 'Extracurricular', 'Sports', 'Music', 'Volunteering' ]
        
        categorical_columns = nominal_columns + ordinal_columns 
        
        ordinal_columns_values = [
                                     ['None', 'High School', 'Some College', "Bachelor's", 'Higher'],  # ParentalEducation 
                                    ['None', 'Low', 'Moderate', 'High', 'Very High'] , 
                ]
        
        #Creation of cleaning transformers
        #-----------------------------------------
        cleaning_transformer = ColumnTransformer( [
                 ("categorical_imputer" , SimpleImputer(strategy= "most_frequent") , categorical_columns ) ,
                ("numerical_imputer" , SimpleImputer(strategy="mean") , numerical_columns) 
                                                ])


        #creation of restore name transformer
        #------------------------------------------------
        restore_name_transformer =  RestoreNameTransformer(categorical_columns + numerical_columns)
       
       
       #creatopm of feature engineering transformer
       #-------------------------------------------------
        feature_engineering_transformer = FeatureEngineeringTransformer()

        
        #creation of preprocessing transformer 
        #-------------------------------------------------


        new_num_columns = ["extra_activities_studies" , "extra_activities_no_studies"]

        numerical_columns = ['Age', 'StudyTimeWeekly', 'Absences'] + new_num_columns

        nominal_columns = [ "Gender" ,"Ethnicity" ]

        ordinal_columns = [ 'ParentalEducation',  'ParentalSupport']


        cat_nominal_preprocessing_steps = Pipeline(
                                            steps = [
                                             ("one_hot_encoder_steps" , OneHotEncoder())
                                                ]   )

        cat_ordinal_preprocessing_steps = Pipeline(
                                         steps =  [
                            ("ordinal_encoder_steps" , OrdinalEncoder(categories =  ordinal_columns_values )  ) , 
                            ("ordinal_scaler" , StandardScaler())
                                        ] )
        
        num_preprocessing_steps = Pipeline(
                                steps = [
                            ("standard_scaler_steps" , StandardScaler())
                                        ] )
        
        preprocessor_transformer = ColumnTransformer(
                                        [
                            ("cat_nominal_preprocessor" ,cat_nominal_preprocessing_steps ,nominal_columns  ), 
                            ("cat_ordinal_preprocessor",   cat_ordinal_preprocessing_steps, ordinal_columns),
                            ("num_preprocessor" , num_preprocessing_steps , numerical_columns )
                                        ])

        #joining all the inputs steps in a pipeline
        input_preprocessor = Pipeline(
             steps = [
                              ("cleaning transformer" , cleaning_transformer) ,
                              ("restore name transformer", restore_name_transformer),
                              ( "feature engineering transformer", feature_engineering_transformer  ) , 
                              ("preprocessor transformer" ,preprocessor_transformer )
                 ]  )
        
        #creating the output preprocessor

        label_values = [["F" , "D" , "C" , "B" , "A"]]
        ouput_encoder = OrdinalEncoder(categories=label_values)

        return input_preprocessor , ouput_encoder

    def initiate_data_transformation(self , train_df:pd.DataFrame , test_df:pd.DataFrame):
        logging.info("enter to the intitiate data transformation funcion")
        try:
            logging.info("transforming boolean column in  numeric columns")
            bool_columns = train_df.select_dtypes(include = ["bool"]).columns
            train_df[bool_columns]  = train_df[bool_columns].astype(int)
            
            bool_columns = test_df.select_dtypes(include = ["bool"]).columns
            test_df[bool_columns]  = test_df[bool_columns].astype(int)

            logging.info("creating preprocessor object")
            
            input_preprocessor , output_encoder = self.get_preprocessor()
            
            label = "GradeClass"
            
            train_input_raw = train_df.drop(columns =[label] )
            train_label_raw = train_df[[label]]

            test_input_raw = test_df.drop(columns = [label])
            test_label_raw = test_df[[label]]

            logging.info("applying preprocessing to the dataset")

            train_input_array = input_preprocessor.fit_transform(train_input_raw)
            test_input_array = input_preprocessor.transform(test_input_raw)

            train_label_array = output_encoder.fit_transform(train_label_raw)
            test_label_array = output_encoder.transform(test_label_raw)

            logging.info("joining features and labels column")

            train_array = np.hstack(( train_input_array,train_label_array ))
            test_array = np.hstack((test_input_array , test_label_array))

            logging.info("saving input and labels preprocessor")

            save_object(file_path=self.dataconfig.input_preprocessor_path , obj=input_preprocessor)
            save_object(file_path=self.dataconfig.output_encoder_path , obj = output_encoder  )

            return (train_array , test_array)
             
        except Exception as e:
            raise CustomException(e , sys)