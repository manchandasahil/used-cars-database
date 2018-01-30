# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 22:17:08 2017

@author: Sahil Manchanda
"""
import pandas as pd
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import to_categorical


cars = pd.read_csv('autos.csv', encoding = "ISO-8859-1")
cars = cars.dropna()
cars = cars[cars["price"]>100]
cars = cars[cars["price"]<35000]
X = cars[["powerPS","kilometer","yearOfRegistration"]]

from sklearn.preprocessing import LabelEncoder
OHE = LabelEncoder()

fuel         = OHE.fit_transform(cars["fuelType"]).reshape(cars.shape[0],1)
vehicleType  = OHE.fit_transform(cars["vehicleType"]).reshape(fuel.shape[0],1)
gearbox      = OHE.fit_transform(cars["gearbox"]).reshape(fuel.shape[0],1)
brand        = OHE.fit_transform(cars["brand"]).reshape(fuel.shape[0],1)
model_car    = OHE.fit_transform(cars["model"]).reshape(fuel.shape[0],1)
repaired     = OHE.fit_transform(cars["notRepairedDamage"]).reshape(fuel.shape[0],1)

#X = np.concatenate((X,brand),axis = 1)
cars["kilometer"] = cars["kilometer"]/1000
cars["yearOfRegistration"] = cars["yearOfRegistration"]-1910
cars["powerPS"] = cars["powerPS"]/100

import numpy as np
X = np.concatenate((X,fuel),axis = 1)
X = np.concatenate((X,vehicleType),axis = 1)
X = np.concatenate((X,gearbox),axis = 1)
X = np.concatenate((X,brand),axis = 1)
X = np.concatenate((X,model_car),axis = 1)
X = np.concatenate((X,repaired),axis = 1)

Y = cars["price"].values
        
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.25,random_state=42)

model = Sequential()
model.add(Dense(80,input_dim = 9,kernel_initializer="normal",activation = 'relu'))
model.add(Dense(20,kernel_initializer="normal",activation = 'relu'))
model.add(Dense(5,kernel_initializer="normal",activation = 'relu'))
model.add(Dense(1,kernel_initializer="normal"))

model.compile(loss='mae',
              optimizer='adam',
              metrics=['mae'])

model.fit(x_train,y_train,epochs=20, verbose=1)
print(model.evaluate(x_test,y_test))

