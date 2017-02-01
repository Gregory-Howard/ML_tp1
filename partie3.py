import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

##############################################################################################
#              Apprentissage non supervisé : Détection d’anomalies                           #
##############################################################################################

# Chargement des données et préparation
df=pd.read_csv('mouse-synthetic-data.txt', sep=' ')
values = df.values
# affichage des données 



# isolation forest
from sklearn.ensemble import IsolationForest



# fit the model
clf = IsolationForest(max_samples=100)
clf.fit(values)
y_pred_train = clf.predict(values)

# plot the line, the samples, and the nearest vectors to the plane
xx, yy = np.meshgrid(np.linspace(0, 1, 50), np.linspace(0, 1, 50))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("IsolationForest")
plt.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
# 1 = considérées normales, -1 = considérées anormales
ok = y_pred_train==1
pas_ok = y_pred_train==-1

b1 = plt.scatter(values[ok, 0], values[ok, 1], c='white')
b2 = plt.scatter(values[pas_ok, 0], values[pas_ok, 1], c='red')

plt.axis('tight')
plt.xlim((0, 1))
plt.ylim((0, 1))
plt.legend([b1,b2],
           ["valid","not valid"],
           loc="upper left")
plt.show()


# PARTIE 2

# Jeu de donnée SMS
