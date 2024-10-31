from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

data_ingestion = DataIngestion()
data_transformation = DataTransformation()
model_trainer = ModelTrainer()

print("initiate data ingestion")
train_df , test_df = data_ingestion.initiate_data_ingestion()

print("initiate data transformation")
train_array , test_array = data_transformation.initiate_data_transformation(train_df , test_df )

print("initiate model training")
best_model_name , best_model_score , best_model = model_trainer.initate_model_training(train_array , test_array)

print(f"reporte de modelos")
print(model_trainer.show_report())
print()

print(f"el mejor modelo fue {best_model_name} con score de {best_model_score}")

print("fin del pipeline de entrenamiento")