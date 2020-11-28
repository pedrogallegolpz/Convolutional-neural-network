# -*- coding: utf-8 -*-

#########################################################################
############ CARGAR LAS LIBRERÍAS NECESARIAS ############################
#########################################################################

# En caso de necesitar instalar keras en google colab,
# ejecutar la siguiente línea:
# !pip install -q keras
# Importar librerías necesarias
import numpy as np
import keras 
import matplotlib.pyplot as plt
import keras.utils as np_utils

# Importar modelos y capas que se van a usar
# A completar
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, UpSampling2D
from keras.layers import BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

# Importar cross validation
from sklearn.model_selection import KFold
#from sklearn.model_selection import StratifiedKFold

# Importar el optimizador a usar
from keras.optimizers import SGD

# Importar el conjunto de datos
from keras.datasets import cifar100

#########################################################################
######## FUNCIÓN PARA CARGAR Y MODIFICAR EL CONJUNTO DE DATOS ###########
#########################################################################

#########
#       VARIABLES GLOBALES
#########
EPOCHS = 200
PATIENCE = 20   # para el early stopping
SPLIT = 0.1
BATCH_SIZE = 64 #valores posibles 32,64, 128



# funcion de parada de codigo
def wait():
    input("Pulsa una tecla para continuar")









# A esta función solo se la llama una vez. Devuelve 4 
# vectores conteniendo, por este orden, las imágenes
# de entrenamiento, las clases de las imágenes de
# entrenamiento, las imágenes del conjunto de test y
# las clases del conjunto de test.
def cargarImagenes():
    # Cargamos Cifar100. Cada imagen tiene tamaño
    # (32 , 32, 3). Nos vamos a quedar con las
    # imágenes de 25 de las clases.
    (x_train, y_train), (x_test, y_test) = cifar100.load_data (label_mode ='fine')
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    train_idx = np.isin(y_train, np.arange(25))
    train_idx = np.reshape (train_idx, -1)
    x_train = x_train[train_idx]
    y_train = y_train[train_idx]
    test_idx = np.isin(y_test, np.arange(25))
    test_idx = np.reshape(test_idx, -1)
    x_test = x_test[test_idx]
    y_test = y_test[test_idx]
    
    # Transformamos los vectores de clases en matrices.
    # Cada componente se convierte en un vector de ceros
    # con un uno en la componente correspondiente a la
    # clase a la que pertenece la imagen. Este paso es
    # necesario para la clasificación multiclase en keras.
    y_train = np_utils.to_categorical(y_train, 25)
    y_test = np_utils.to_categorical(y_test, 25)
    
    return x_train , y_train , x_test , y_test

#########################################################################
######## FUNCIÓN PARA OBTENER EL ACCURACY DEL CONJUNTO DE TEST ##########
#########################################################################

# Esta función devuelve la accuracy de un modelo, 
# definida como el porcentaje de etiquetas bien predichas
# frente al total de etiquetas. Como parámetros es
# necesario pasarle el vector de etiquetas verdaderas
# y el vector de etiquetas predichas, en el formato de
# keras (matrices donde cada etiqueta ocupa una fila,
# con un 1 en la posición de la clase a la que pertenece y un 0 en las demás).
def calcularAccuracy(labels, preds):
    labels = np.argmax(labels, axis = 1)
    preds = np.argmax(preds, axis = 1)
    accuracy = sum(labels == preds)/len(labels)
    return accuracy

#########################################################################
## FUNCIÓN PARA PINTAR LA PÉRDIDA Y EL ACCURACY EN TRAIN Y VALIDACIÓN ###
#########################################################################

# Esta función pinta dos gráficas, una con la evolución
# de la función de pérdida en el conjunto de train y
# en el de validación, y otra con la evolución de la
# accuracy en el conjunto de train y el de validación.
# Es necesario pasarle como parámetro el historial del
# entrenamiento del modelo (lo que devuelven las
# funciones fit() y fit_generator()).
def mostrarEvolucion(hist):
    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    plt.plot(loss)
    plt.plot(val_loss)
    plt.legend(['Training loss', 'Validation loss'])
    plt.show()
    
    #print(hist.history.keys())
    
    acc = hist.history['accuracy']
    val_acc = hist.history['val_accuracy']
    plt.plot(acc)
    plt.plot(val_acc)
    plt.legend(['Training accuracy','Validation accuracy'])
    plt.show()

#########################################################################
################## DEFINICIÓN DEL MODELO BASENET ########################
#########################################################################
# A completar
def modelBaseNet():
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5),activation='relu',input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    model.add(Conv2D(16, kernel_size=(5, 5),activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))

    model.add(Flatten())
    model.add(Dense(50, activation='relu'))
    model.add(Dense(25, activation='softmax'))

    return model
    
#########################################################################
######### DEFINICIÓN DEL OPTIMIZADOR Y COMPILACIÓN DEL MODELO ###########
#########################################################################
# A completar
def optAndComp(model):
    # Definimos el optimizador
    opt = SGD(lr = 0.01, decay = 1e-6, momentum = 0.9, nesterov = True)
    
    # Compilamos el modelo
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=opt, metrics=['accuracy'])
    return model


# Una vez tenemos el modelo base, y antes de entrenar, vamos a guardar los
# pesos aleatorios con los que empieza la red, para poder reestablecerlos
# después y comparar resultados entre no usar mejoras y sí usarlas.
weights = optAndComp(modelBaseNet()).get_weights()

#########################################################################
###################### ENTRENAMIENTO DEL MODELO #########################
#########################################################################
# A completar
def samplesGenerator(x_train , y_train , x_test , y_test, mejorado=True):
    
    if(mejorado):
        datagen = ImageDataGenerator(featurewise_center = True,
                                 featurewise_std_normalization = True,
                                 width_shift_range = 0.05,     # desplazamiento horizontal
                                 height_shift_range = 0.1,    # desplazamiento vertical
                                 horizontal_flip = True,      # volteo horizontal
                                 zoom_range = 0.2,
                                 validation_split = SPLIT)
         
        # Se estandarizan los datos. 
        # Con fit aplicamos todo a X_train 
        # Con standardize aplicamos solo la media=0 y la varianza=1
        datagen.fit(x_train)
        datagen.standardize(x_test)
    else:
        datagen = ImageDataGenerator(validation_split = 0.1)
    
   
    
    # Definimos los conjuntos de entrenamiento y de test
    training = datagen.flow(x_train,
                 y_train,
                 batch_size = BATCH_SIZE,
                 subset = 'training')
    
    
    validacion = datagen.flow(x_train,
                 y_train,
                 batch_size = BATCH_SIZE,
                 subset = 'validation')
    
    
    return training , validacion, x_test , y_test
   

# A parte de entrenar nuestro modelo, estandarizamos los conjuntos de ser necesario
def training(model, x_train, y_train, x_test , y_test, mejorado=True):
    # Cogemos nuestros conjuntos normalizados y listos para poder entrenarlos y evaluarlos
    training , validacion, x_test , y_test = samplesGenerator(x_train, y_train, x_test , y_test, mejorado)
    
    # Definimos el early stopping
    # Callbacks 
    callbacks_stops = []
    
    # Early stopping para loss y accuracy
    early_stopping_loss = EarlyStopping(monitor = 'val_loss',
                                        patience = PATIENCE,
                                        restore_best_weights = True)
    early_stopping_val = EarlyStopping(monitor = 'val_accuracy',
                                       patience = PATIENCE,
                                       restore_best_weights = True)
    
    callbacks_stops.append(early_stopping_loss)
    callbacks_stops.append(early_stopping_val)
    
    # Entrenamos el modelo
    history = model.fit(training,
                        epochs = EPOCHS,
                        steps_per_epoch = len(x_train)*(1-SPLIT)/BATCH_SIZE,
                        verbose = 1,
                        validation_data = validacion,
                        validation_steps = len(x_train)*SPLIT/BATCH_SIZE,
                        callbacks = callbacks_stops)    
    
    return model, history   
 
    

#########################################################################
################ PREDICCIÓN SOBRE EL CONJUNTO DE TEST ###################
#########################################################################
# A completar
def prediction(model, x_test , y_test):
     # Evaluamos el modelo con los test
    sc=model.evaluate(x_test,y_test)
    loss=sc[0]
    score=sc[1]
  
    print("Pérdida media al evaluar:"+str(loss))
    print("Predicción media al evaluar:"+str(score)) 
        
    return loss, score

#########################################################################
########################## MEJORA DEL MODELO ############################
#########################################################################

# A completar. Tanto la normalización de los datos como el data
# augmentation debe hacerse con la clase ImageDataGenerator.
# Se recomienda ir entrenando con cada paso para comprobar
# en qué grado mejora cada uno de ellos.

############## DEFINICIÓN DEL MODELO BASENET MEJORADO ###################
# A completar
def modelBaseNetMejorado():
    
    model = Sequential()
    model.add(Conv2D(6, kernel_size=(5, 5),input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Dropout(0.1))

    model.add(Conv2D(16, kernel_size=(5, 5)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=None, padding='valid'))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(50))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(25, activation='softmax'))
    
    return model
   
    
    
def profundidadBonus():
    model = Sequential()

    # 32x32x3 ENTRADA

    model.add(Conv2D(32,
                     padding = 'same',
                     kernel_size=(3, 3),
                     input_shape=(32,32,3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 32x32x32

    model.add(Conv2D(64,
                     kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 30x30x64

    model.add(Conv2D(64, 
                     kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    # 14x14x64
  
    model.add(Conv2D(128, 
                     padding = 'same',
                     kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 14x14x128

    model.add(Conv2D(128, 
                     kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.35))

    # 12x12x128

    model.add(Conv2D(256, 
                     padding = 'same',
                     kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    # 12x12x256
  
    model.add(Conv2D(256, 
                     padding = 'same',
                     kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.35))

    # 6x6x256

    model.add(Conv2D(256,
                     padding = 'same', 
                     kernel_size=(3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(AveragePooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.55))

    # 3x3x256
    
    model.add(Flatten())

    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 

    
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))    
    model.add(Dropout(0.7))
    
    model.add(Dense(25, activation='softmax'))

    return model


################## ENTRENAMIENTO DEL MODELO MEJORADO ######################

# Definición de los conjuntos con las mejoras en el samplesGenerator
"""
hecho en la funcion de arriba de samplesGenerator
"""




#########################################################################
#########################################################################
############################### PROGRAMA ################################
#########################################################################
#########################################################################
def ejecutaRed():
    # Cargamos los conjuntos 
    x_train , y_train, x_test , y_test = cargarImagenes()   
    x_train2 , y_train2, x_test2 , y_test2 = cargarImagenes()   
    x_train3 , y_train3, x_test3 , y_test3 = cargarImagenes()   
    
    print("Se va a proceder a ejecutar la práctica. Se ejecutarán de forma casi paralela 3 versiones distintas:")
    print("-El modelo BaseNet")
    print("-El modelo BaseNet con mejoras de dropout y batchnormalization")
    print("-El modelo con mayor profundidad donde se ha incluído directamente el bonus, como dice la memoria")
    print("\nSe pondrán puntos de parada a partir del apartado de mostrar la evolución del modelo. Primero entrenaremos los modelos y luego se mostrará su evolución")
    wait()
    
    # Cogemos el modelo BaseNet y BaseNetMejorado
    print("Asignando los modelos...")
    base_net = modelBaseNet()    
    base_net_mejorado = modelBaseNetMejorado()
    bonus = profundidadBonus()
    
    
    # Compilamos el modelo
    print("Compilando los modelos...")
    base_net_optcomp = optAndComp(base_net)
    base_net_optcomp_mejorado = optAndComp(base_net_mejorado)
    bonus_optcomp = optAndComp(bonus)
    
    # Entrenamos el modelo
    print("Entrenando los modelos...")
    print("Entrenamos modelo BaseNet")
    model, history=training(base_net_optcomp, x_train , y_train , x_test , y_test, mejorado = False)
    print("\n")
    print("Entrenamos modelo BaseNet con mejoras dropout y batchnormalization")
    model_mejorado, history_mejorado=training(base_net_optcomp_mejorado, x_train2 , y_train2 , x_test2 , y_test2, mejorado = True)
    print("\n")
    print("Entrenamos  modelo con mayor profundidad donde se ha incluído directamente el bonus")
    model_bonus, history_bonus = training(bonus_optcomp, x_train3, y_train3, x_test3, y_test3, mejorado = True)

    # Mostramos la evolución
    print("Vamos a proceder a mostrar las evoluciones")
    print("Evolución del entrenamiento del modelo BaseNet")
    mostrarEvolucion(history)
    wait()
    print("Evolución del entrenamiento del modelo con mejoras BatchNormalization y Dropout")
    mostrarEvolucion(history_mejorado)
    wait()
    print("Evolución del entrenamiento del modelo con profundidad y BONUS")
    mostrarEvolucion(history_bonus)
    wait()
    
    
    # Predicción
    print("Predicciones de:")
    print("BaseNet:")
    prediction(model, x_test , y_test)
    print("BaseNetMejorado")
    prediction(model_mejorado, x_test2 , y_test2)
    print("Bonus")
    prediction(model_bonus, x_test3 , y_test3)



######
# EJECUCIÓN
######
ejecutaRed()