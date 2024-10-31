import sys
from src.exception import CustomException
from flask import Flask , request ,render_template
from src.pipelines.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict" , methods =["GET" , "POST"] )
def predict_datapoint():
    if request.method =="GET":
        return render_template("predict.html")
    else:
        try:
            user_data =  CustomData(
                Age = request.form.get("Age"),
                Gender= request.form.get("Gender"),
                Ethnicity= request.form.get("Ethnicity"),
                ParentalEducation= request.form.get("ParentalEducation"),
                StudyTimeWeekly= request.form.get("StudyTimeWeekly"),
                Absences= request.form.get("Absences"),
                Tutoring= request.form.get("Tutoring"),
                ParentalSupport= request.form.get("ParentalSupport"),
                Extracurricular= request.form.get("Extracurricular"),
                Sports= request.form.get("Sports"),
                Music= request.form.get("Music"),
                Volunteering= request.form.get("Volunteering")
            )
            user_data_df = user_data.get_data_as_dataframe()
            #print(user_data_df)
            predict_pipeline = PredictPipeline()
            preds = predict_pipeline.predict(user_data_df)
            #print(preds)
            return render_template("predict.html", results = preds[0][0])

        except Exception as e:
            raise CustomException(e , sys)
        
if __name__ =="__main__":
    app.run(host = "0.0.0.0" , port = 8080 , debug = True)