import pandas as pd
from src.utils import *
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainerConfig
from src.exception import CustomException
class CustomData():
    def __init__(self,Age,Gender,Ethnicity,ParentalEducation,
                 StudyTimeWeekly,Absences,Tutoring,ParentalSupport,Extracurricular,Sports,Music,Volunteering ):
        self.user_data = pd.DataFrame({
            "Age":[Age] , 
            "Gender": [Gender], 
            "Ethnicity":[Ethnicity],
            "ParentalEducation":[ParentalEducation],
            "StudyTimeWeekly":[StudyTimeWeekly],
            "Absences":[Absences] ,
            "Tutoring":[int(bool(Tutoring))],
            "ParentalSupport":[ParentalSupport] , 
            "Extracurricular": [int(bool(Extracurricular))], 
            "Sports":[int(bool(Sports))] , 
            "Music": [int(bool(Music))], 
            "Volunteering":[int(bool(Volunteering))] 
        })
    def get_data_as_dataframe(self):
        return self.user_data

class PredictPipeline():
    def __init__(self):
        self.input_preprocessor = load_object(DataTransformationConfig.input_preprocessor_path)
        self.output_encoder = load_object(DataTransformationConfig.output_encoder_path)
        self.model = load_object(ModelTrainerConfig.model_path)
    
    def predict(self , user_data):
        try:
            data_transformed = self.input_preprocessor.transform(user_data)
            preds = self.model.predict(data_transformed)

            preds =  self.output_encoder.inverse_transform([preds])
            return preds
        except Exception as e:
            raise CustomException(e,  sys)
