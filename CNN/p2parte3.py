# -*- coding: utf-8 -*-
"""EsquemaParte3.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Ge_Pw-txgQ1PRJTdJPioBOl02yILeVuf
"""

#########################################################################
################### OBTENER LA BASE DE DATOS ############################
#########################################################################

# Descargar las imágenes de http://www.vision.caltech.edu/visipedia/CUB-200.html
# Descomprimir el fichero.
# Descargar también el fichero list.tar.gz, descomprimirlo y guardar los ficheros
# test.txt y train.txt dentro de la carpeta de imágenes anterior. Estos 
# dos ficheros contienen la partición en train y test del conjunto de datos.

##### EN CASO DE USAR COLABORATORY
# Sube tanto las imágenes como los ficheros text.txt y train.txt a tu drive.
# Después, ejecuta esta celda y sigue las instrucciones para montar 
# tu drive en colaboratory.
"""
from google.colab import drive
drive.mount('/content/drive')
"""

#########################################################################
################ CARGAR LAS LIBRERÍAS NECESARIAS ########################
#########################################################################

# Terminar de rellenar este bloque con lo que vaya haciendo falta

# Importar librerías necesarias
import numpy as np
import keras
import keras.utils as np_utils
from keras.preprocessing.image import load_img,img_to_array
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Importar modelos y capas específicas que se van a usar
from keras.models import  Model, Sequential
from keras.layers import Conv2D, Dense
from keras.layers import Flatten, Dropout, BatchNormalization

# Importar el modelo ResNet50 y su respectiva función de preprocesamiento,
# que es necesario pasarle a las imágenes para usar este modelo
from keras.applications.resnet import ResNet50, preprocess_input


# Importar el optimizador a usar
from keras.optimizers import SGD



# VARIABLES GLOBALES
#PATH = "./imagenes/"
PATH = "/content/drive/My Drive/imagenes"
BATCH_SIZE = 64
SPLIT = 0.1


# funcion de parada de codigo
def wait():
    input("Pulsa una tecla para continuar")

#########################################################################
################## FUNCIÓN PARA LEER LAS IMÁGENES #######################
#########################################################################

# Dado un fichero train.txt o test.txt y el path donde se encuentran los
# ficheros y las imágenes, esta función lee las imágenes
# especificadas en ese fichero y devuelve las imágenes en un vector y 
# sus clases en otro.

def leerImagenes(vec_imagenes, path):
  clases = np.array([img.split('/')[0] for img in vec_imagenes])
  imagenes = np.array([img_to_array(load_img(path + "/" + img, 
                                             target_size = (224, 224))) 
                       for img in vec_imagenes])
  return imagenes, clases

#########################################################################
############# FUNCIÓN PARA CARGAR EL CONJUNTO DE DATOS ##################
#########################################################################

# Usando la función anterior, y dado el path donde se encuentran las
# imágenes y los archivos "train.txt" y "test.txt", devuelve las 
# imágenes y las clases de train y test para usarlas con keras
# directamente.

def cargarDatos(path):
  # Cargamos los ficheros
  train_images = np.loadtxt(path + "/train.txt", dtype = str)
  test_images = np.loadtxt(path + "/test.txt", dtype = str)
  
  # Leemos las imágenes con la función anterior
  train, train_clases = leerImagenes(train_images, path)
  test, test_clases = leerImagenes(test_images, path)
  
  # Pasamos los vectores de las clases a matrices 
  # Para ello, primero pasamos las clases a números enteros
  clases_posibles = np.unique(np.copy(train_clases))
  for i in range(len(clases_posibles)):
    train_clases[train_clases == clases_posibles[i]] = i
    test_clases[test_clases == clases_posibles[i]] = i

  # Después, usamos la función to_categorical()
  train_clases = np_utils.to_categorical(train_clases, 200)
  test_clases = np_utils.to_categorical(test_clases, 200)
  
  # Barajar los datos
  train_perm = np.random.permutation(len(train))
  train = train[train_perm]
  train_clases = train_clases[train_perm]

  test_perm = np.random.permutation(len(test))
  test = test[test_perm]
  test_clases = test_clases[test_perm]
  
  return train, train_clases, test, test_clases

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve el accuracy de un modelo, definido como el 
# porcentaje de etiquetas bien predichas frente al total de etiquetas.
# Como parámetros es necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de keras (matrices
# donde cada etiqueta ocupa una fila, con un 1 en la posición de la clase
# a la que pertenece y 0 en las demás).

def calcularAccuracy(labels, preds):
  labels = np.argmax(labels, axis = 1)
  preds = np.argmax(preds, axis = 1)
  
  accuracy = sum(labels == preds)/len(labels)
  
  return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución de la función
# de pérdida en el conjunto de train y en el de validación, y otra
# con la evolución del accuracy en el conjunto de train y en el de
# validación. Es necesario pasarle como parámetro el historial
# del entrenamiento del modelo (lo que devuelven las funciones
# fit() y fit_generator()).

def mostrarEvolucion(hist):

  loss = hist.history['loss']
  val_loss = hist.history['val_loss']
  plt.plot(loss)
  plt.plot(val_loss)
  plt.legend(['Training loss', 'Validation loss'])
  plt.show()

  acc = hist.history['accuracy']
  val_acc = hist.history['val_accuracy']
  plt.plot(acc)
  plt.plot(val_acc)
  plt.legend(['Training accuracy', 'Validation accuracy'])
  plt.show()

"""## Usar ResNet50 preentrenada en ImageNet como un extractor de características"""

def extractorCaracteristicas(path, experimento):
    """
    Primera parte del apartado 3.
    -experimento: variable que nos indicará si estamos en el experimento 1 con 
    modelo básico, 2 con modelo con más FC, <0 and >2 con capas conv
    """
    
    # Completamos la red
    if(experimento == 0):
        modelo = modelFCBasic()
        pool = 'avg'  # Caracteristicas de tamaño 2048
    elif(experimento == 1):
        modelo = modelFC()
        pool = 'avg'  # Caracteristicas de tamaño 2048
    else:
        modelo = modelConv()
        pool = None   # Caracteristicas de tamaño 7x7x2048


    #Cargamos los datos
    x_train, y_train, x_test, y_test = cargarDatos(path)
    
    # Definir un objeto de la clase ImageDataGenerator para train y otro para test
    # con sus respectivos argumentos.
    # A completar
    datagen_train = ImageDataGenerator(preprocessing_function = preprocess_input)
    datagen_test = ImageDataGenerator(preprocessing_function = preprocess_input)
    
    # Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
    # A completar    
    resnet50 = ResNet50(include_top = False, 
                        weights = 'imagenet',
                        pooling = pool)    #con esta opción la salida es de 2048 entradas, ahorramos el flatten
    

    # Extraer las características las imágenes con el modelo anterior.
    # A completar
    print("Extraemos las características")
    caracteristicas_train = resnet50.predict_generator(datagen_train.flow(x_train,
                                                                          batch_size = 1,
                                                                          shuffle = False),
                                                        verbose = 1,
                                                        steps = len(x_train))
    
    caracteristicas_test = resnet50.predict_generator(datagen_test.flow(x_test,
                                                                          batch_size = 1,
                                                                          shuffle = False),
                                                        verbose = 1,
                                                        steps = len(x_test))
    
    
    # Compilamos el modelo
    opt = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    modelo.compile(loss = keras.losses.categorical_crossentropy,
                   optimizer = opt,
                   metrics = ['accuracy'])
    
    # Entrenamos el modelo
    history = modelo.fit(caracteristicas_train, y_train,
                         batch_size = BATCH_SIZE,
                         epochs = 40, 
                         verbose = 1,
                         validation_split = SPLIT)
    mostrarEvolucion(history)
    
    
    # Evaluamos el modelo
    sc = modelo.evaluate(caracteristicas_test, y_test, verbose = 0)
    
    loss=sc[0]
    score=sc[1]
  
    print("Pérdida media al evaluar:"+str(loss))
    print("Predicción media al evaluar:"+str(score)) 
    
    return history, sc
    
    

# Las características extraídas en el paso anterior van a ser la entrada
# de un pequeño modelo de dos capas Fully Conected, donde la última será la que 
# nos clasifique las clases de Caltech-UCSD (200 clases). De esta forma, es 
# como si hubiéramos fijado todos los parámetros de ResNet50 y estuviésemos
# entrenando únicamente las capas añadidas. Definir dicho modelo.
# A completar: definición del modelo, del optimizador y compilación y
# entrenamiento del modelo.
# En la función fit() puedes usar el argumento validation_split
def modelConv():
    model = Sequential()

    model.add(Conv2D(1024,
                     kernel_size = (3, 3),
                     activation = 'relu',
                     input_shape = (7,7,2048,)))
    model.add(BatchNormalization())
    model.add(Conv2D(1024,
                     kernel_size = (3,3),
                     activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(1024, 
                    activation = 'relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))

    model.add(Dense(200,
                    activation = 'softmax'))
    
    return model

def modelFC():
    model = Sequential()
    
    model.add(Dense(1024,
                     activation = 'relu',
                     input_shape = (2048,)))
    model.add(Dense(512,
                     activation = 'relu'))
    model.add(Dropout(0.7))
    model.add(Dense(200,
                     activation = 'softmax'))
    
    return model

def modelFCBasic():
    """
    Crea un modelo básico FC
    """
    model = Sequential()
    model.add(Dense(200,
                    activation = 'softmax'))
    
    return model
    

"""## Reentrenar ResNet50 (fine tunning)"""

# Definir un objeto de la clase ImageDataGenerator para train y otro para test
# con sus respectivos argumentos.
# A completar
def fineTunning(path):
    #Cargamos los datos
    x_train, y_train, x_test, y_test = cargarDatos(path)
    
    # Definir un objeto de la clase ImageDataGenerator para train y otro para test
    # con sus respectivos argumentos.
    # A completar
    datagen_train = ImageDataGenerator(preprocessing_function = preprocess_input,
                                       validation_split = SPLIT)
    datagen_test = ImageDataGenerator(preprocessing_function = preprocess_input)
    
    # Definir el modelo ResNet50 (preentrenado en ImageNet y sin la última capa).
    # A completar    
    resnet50 = ResNet50(include_top = False, 
                        weights = 'imagenet',
                        pooling = 'avg',
                        input_shape = (224,224,3))   #con esta opción la salida es de 2048 entradas
                            
    
    # Vemos que la capa dense no está
    print(resnet50.summary())


    # Definimos nuestro modelo entero
    salida = modelFC_fineTunning(resnet50.output)
    model = Model(inputs = resnet50.input, outputs = salida)
    
    # Compilamos el modelo
    opt = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    model.compile(loss = keras.losses.categorical_crossentropy,
                  optimizer = opt,
                  metrics = ['accuracy'])
    
    # Entrenamos el modelo
    history = model.fit(datagen_train.flow(x_train, 
                                           y_train,
                                           batch_size = BATCH_SIZE,
                                           subset = 'training'),
                         epochs = 20, 
                         steps_per_epoch = len(x_train)*(1-SPLIT)/BATCH_SIZE,
                         verbose = 1,
                         validation_data = datagen_train.flow(x_train,
                                                              y_train,
                                                              batch_size = BATCH_SIZE,
                                                              subset = 'validation'),
                         validation_steps = len(x_train)*SPLIT / BATCH_SIZE)

                         
    mostrarEvolucion(history)
    
    
    # Evaluamos el modelo
    sc = model.evaluate(datagen_test.flow(x_test,
                                          y_test,
                                          batch_size = BATCH_SIZE,
                                          shuffle = False),
                        verbose = 0,
                        steps = len(x_test))
    loss=sc[0]
    score=sc[1]
  
    print("Pérdida media al evaluar:"+str(loss))
    print("Predicción media al evaluar:"+str(score)) 
    
    return history, sc

# Añadir nuevas capas al final de ResNet50 (recuerda que es una instancia de
# la clase Model).
def modelFC_fineTunning(modelo_actual):
    modelo_actual = Dense(2048, activation = 'relu') (modelo_actual) 
    modelo_actual = Dropout(0.7) (modelo_actual)
    salida = Dense(200, activation = 'softmax') (modelo_actual)
    
    return salida









###############################
############ MAIN #############
###############################
print("Se va a ejecutar el apartado 3. Se ejecutará en este orden:")
print("-El extractor de características con un modelo básico que contiene solo la última capa FC")
print("-El extractor de características con un modelo que contiene varias capas FC")
print("-El extractor de características con un modelo que contiene ademaś capas convolucionales")
print("-El segundo ejercicio que propone el apartado 3")
print("Empezamos...")

print("-El extractor de características con un modelo básico que contiene solo la última capa FC")
extractorCaracteristicas(PATH,0)
wait()
print("-El extractor de características con un modelo que contiene varias capas FC")
extractorCaracteristicas(PATH,1)
wait()
print("-El extractor de características con un modelo que contiene ademaś capas convolucionales")
extractorCaracteristicas(PATH,2)
wait()
print("Se va a ejecutar el segundo ejercicio del apartado 3.")
wait()
print("Se va a reentrenar la red entera con nuestro modelo añadido")
fineTunning(PATH)
    