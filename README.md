# practicaml-005

en este proyecto voy a crear un modelo que realice la prediccion de notas de estudiantes en base a sus caracteristicas<br>

url del dataset :  https://www.kaggle.com/datasets/rabieelkharoua/students-performance-dataset   <br>

# aspectos finales del proyecto <br> 
- nombre del mejor modelo : XGBClaassifer
- metrica que se utilizo: accuracy
- rendimiento del mejor modelo:  0.699374

inconvenientes encontrados durante el desarrollo del proyecto: <br>
- debido a que el gpa y las class label estan relacionadas de manera logica, elimine la columna del gpa, para predecir la classe label tomando en cuenta las classes
- no me di cuenta hasta que termine el modeling, pero la cantidad de datos, que habia en el dataset era muy poca 
eran como 3000 filas, y eso es muy poco para alcanzar un rendimiento bueno con los modelos
- la cantidad de datos etiquetados de la clase A , la mejor nota posible eran muy pocas
- las metricas en general con todos los modelos empleados fuerom muy pobres, ninguno supero al 70, lo cual a mi conclusion el modelo fue malo en terminos de rendimiento

<br>
Conclusiones 
- es importante tener una gran cantidad de datos etiquetados para cada posible clase, 
- es importante tener una gran cantidad de datos en general para alcanzar un buen rendimiento con los modelos
