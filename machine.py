#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

from keras import models
from keras import layers
import pandas as pd
import numpy as np
from scipy.stats import lognorm


print("This model trains a Neural Network aiming to predict the productivity! \n")
print("Be patient! This will take a while! \n")

#Importando a base de dados
#Ao alterar a base de treinamento e teste, deve se alterar aqui os arquivos train_data, train_targets, test_data, test_targets

###################################################################################################
#loading the dataset
full_dataset = pd.read_csv('full_dataset.txt', sep=",",header=None)
#splitting the dataset in train data and test data
train_dataset = full_dataset.sample(frac=0.7, random_state=30)
test_dataset = full_dataset.drop(train_dataset.index)
#splitting the dataset in train_data (all columns unless the last one related to Productivity)
train_data =train_dataset.iloc[:,:-1]
#generating the dataset of train_targets (the last column is the productivity)
train_targets=train_dataset.iloc[:,-1:]
#splitting the dataset in test_data (all columns unless the last one related to Productivity)
test_data =test_dataset.iloc[:,:-1]
#generating the dataset of test_targets (the last column is the productivity)
test_targets=test_dataset.iloc[:,-1:]
###################################################################################################


########################################################################################
#Caso queira fazer a quebra das bases de treinamento e teste manualmente segue o link.
#train_data = pd.read_csv('999_train_data_final.txt', sep=",", header=None)
#train_targets = pd.read_csv('999_train_targets_final.txt', sep=",", header=None)
#test_data = pd.read_csv('999_test_data_final.txt', sep=",", header=None)
#test_targets = pd.read_csv('999_test_targets_final.txt', sep=",", header=None)
########################################################################################

#Padronizando os dados (média e desvio padrão)
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std
test_data -= mean
test_data /= std

#Criando a estrutura da rede. Ao alterar a estrutura, numero de neurônios e camadas deve ser alterado aqui!!!
############################################################################################################
def build_model():
  model = models.Sequential()
  model.add(layers.Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(1))
  model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
  return model
############################################################################################################

#Dividindo a amostra em tamanhos iguais para realizar o K-fold
k = 4
num_val_samples = len(train_data) // k
num_epochs = 10
all_scores = []

#Calculando a média da produtividade da base de treinamento
average_prod_train_targets = np.mean(train_targets)
print('Average Productivity of Train Targets:= %f \n' % average_prod_train_targets)

#Realizando o K-Fold
print('Performing the k-fold operation:')
for i in range(k):
  print('Processing fold #', i)
  val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
  val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
  partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0)
  partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
  model = build_model()
  model.fit(partial_train_data, partial_train_targets,epochs=num_epochs, batch_size=1, verbose=0)
  val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
  all_scores.append(val_mae)
print('\n')

# Calculando a média da produtividade da base de treinamento 
average_error_kfold = np.mean(all_scores)
print('Average Error of K-Fold Cross Validation:= %f \n' % average_error_kfold)

# Realizando o K-fold com k
num_epochs = 50
print('Performing the k-fold operation with considering %d epochs' % num_epochs)
all_mae_histories = []
for i in range(k):
  print('Processing fold #', i)
  val_data = train_data[i * num_val_samples: (i + 1) * num_val_samples]
  val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]
  partial_train_data = np.concatenate([train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0)
  partial_train_targets = np.concatenate([train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0)
  model = build_model()
  history = model.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs=num_epochs, batch_size=1, verbose=0)
  mae_history = history.history['val_mae']
  all_mae_histories.append(mae_history)
print('\n')

#Construindo a rede e calculando os dados
print('Building up the model and fitting - minimizing the error')
model = build_model()
model.fit(train_data, train_targets,epochs=100, batch_size=4, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
print('\n')

print('Deep Learining Model Summary \n')
model.summary()

#Calculando o erro quadratico médio mse_score_data
mse_score_data = np.sqrt(test_mse_score)
mae_score_data = np.sqrt(test_mae_score)
print('Mean Squared Error:= %f \n' % mse_score_data)
print('Mean Absolute Error:= %f \n' % mae_score_data)
print('End of Training.')

#Carregar os dados brutos para estimativa de produtividade.

#Alterar o nome do arquivo aqui!!!!
#######################################################################
observ_test = pd.read_csv('04082021_POC_munich2_adjusted.txt', sep=",", header=None)
#######################################################################
print('Raw Data which the productivity must be forecasted: \n')
print(observ_test)
# mean and std (standard deviation are related with train_data)
observ_test -= mean
observ_test /= std

#Calculando o tamanho (numero de linhas) do arquivo que se deseja calcular a produtividade.
size_observ_test = len(observ_test)

#Calculando a produtividade esperada de fato!

#print(model.predict(observ_test[:130]))
expected_prod = model.predict(observ_test[:size_observ_test])
print(expected_prod)
pure_risk = lognorm.pdf(np.mean(expected_prod),np.var(expected_prod),1)
print('Expected Probability:= %f \n'% pure_risk)