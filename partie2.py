import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt

##############################################################################################
# Apprentissage supervisé sur des données textuelles : Feature engineering et Classification #
##############################################################################################

# Chargement des données et préparation
df=pd.read_csv('SMSSpamCollection.data', sep='\t')
values = df.values
