import numpy as np
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import time
warnings.filterwarnings('ignore')
######################
# pas important
def pretty(d, indent=0):
   for key, value in d.iteritems():
      print '\t' * indent + str(key)
      if isinstance(value, dict):
         pretty(value, indent+1)
      else:
         print '\t' * (indent+1) + str(value)

def amelioration(stat,standartStat):
    for i in stat:
        print "Pour l'algo : "+str(i)
        print "Amelioration de l'accuracy par rapport au non standardisé : {:.2f}%\n".format((standartStat[i]["mean"]-stat[i]["mean"])*100)
         
######################
###########################################################
#                     CONSTANTES                          #
###########################################################
INDICES_CONTINUS = [1,2,7,10,13,14]
INDICES_DISCRET = [0,3,4,5,6,8,9,11,12]


###########################################################
#                   PARTIE 1                              #
###########################################################
# Chargement des données et préparation
df=pd.read_csv('credit.data', sep='\t')
values = df.values
'''
caracteristics = values[:,0:(values.shape[1]-1)]
target = values[:,(values.shape[1]-1)]
caracteristics = caracteristics[:,INDICES_CONTINUS]
for i in range(0,caracteristics.shape[0]):
    for j in range(0,caracteristics.shape[1]):
        if isinstance(caracteristics[i][j], basestring) and caracteristics[i][j]=="?":
            caracteristics[i][j]=np.nan

caracteristics=caracteristics.astype(np.float)

# Supression des lignes ayant un nan
indexes= ~np.isnan(caracteristics).any(axis=1)
caracteristics = caracteristics[indexes]
target=target[indexes]
print "Nombre d'elements : "+str(caracteristics.shape[0])
print "Nombre de colonnes : "+str(caracteristics.shape[1])

# binarisation du target
target[target=="+"]=1
target[target=="-"]=0
target=target.astype(np.float)

# affichage de la respartition des classe à prédire
plt.hist(target)
plt.xlabel("classe")
plt.ylabel("nb elements")
plt.show()

# apprentissage et evaluation de modèles
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
import sklearn.ensemble as ensemble
from sklearn.neighbors import KNeighborsClassifier
params = {'n_estimators': 1200, 'max_depth': 3, 'subsample': 0.5, 'learning_rate': 0.01, 'min_samples_leaf': 1, 'random_state': 3}

# Definitions des classifieurs 
clfs = {
'CART': tree.DecisionTreeClassifier(),
'NB': GaussianNB(),
"DecStump":DecisionTreeClassifier(max_depth=1),
"AdaBoost":AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200),
"MLP":MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
"GradientBoosting" : ensemble.GradientBoostingClassifier(**params),
"NearestNeighbors":KNeighborsClassifier(n_neighbors=50),
"RandomForestClassifier":ensemble.RandomForestClassifier()
}



# fonction pour estimer la performance des classifieurs
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
def run_classifiers(clfs,caracteristics, target):
    clfs_stat={}
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    for i in clfs:
        
        clf = clfs[i] #clf correspond au ième algorithme dans votre dictionnaire clfs.
        before = time.time()
        cv_acc = cross_val_score(clf, caracteristics, target, cv=kf) #pour le calcul de l’accuracy
        after = time.time()
        cv_auc = cross_val_score(clf, caracteristics, target, cv=kf, scoring="roc_auc") #pour le calcul de auc
        cv_recall = cross_val_score(clf, caracteristics, target, cv=kf, scoring="recall") #pour le calcul de recall
        cv_precision = cross_val_score(clf, caracteristics, target, cv=kf, scoring="precision") #pour le calcul de recall
        clfs_stat[i] = {"mean":np.mean(cv_acc),"max":np.max(cv_acc),"min":np.min(cv_acc),
                        "std":np.std(cv_acc), "time":after-before,
                        "meanAuc":cv_auc.mean(),"meanRecall":cv_recall.mean(),"meanPrecision":cv_precision.mean()}
    return clfs_stat

clfs_stat = run_classifiers(clfs,caracteristics, target)

### NORMALISATION DES VARIABLES CONTINUES ###
from sklearn.preprocessing import StandardScaler
caracteristics = StandardScaler().fit_transform(caracteristics)

# Lancement des classifieurs et affichage des ameliorations
clfs_stat_Standard = run_classifiers(clfs,caracteristics, target)
amelioration(clfs_stat,clfs_stat_Standard)

### ANALYSE EN COMPOSANTE PRINCIPALE ###
from sklearn.decomposition import PCA

# Transformations des caracteristiques
caracteristics_pca = None
for nbC in range(1, caracteristics.shape[1]):
    pca = PCA(n_components=nbC)
    pca.fit(caracteristics, target)
    if sum(pca.explained_variance_ratio_) >= 0.7:
        print("Keeping {} components out of {}, representing {:.2f}% of the original information".format(nbC, caracteristics.shape[1], sum(pca.explained_variance_ratio_) * 100))
        caracteristics_pca = pca.transform(caracteristics)
        break
# Combinaison des ancienes caracteristiques avec ACP
caracteristics_pca = np.append(caracteristics_pca, caracteristics, axis=1)
# Lancement des classifieurs et affichage des ameliorations
clfs_stat_pca = run_classifiers(clfs,caracteristics_pca, target)
amelioration(clfs_stat_Standard,clfs_stat_pca)



### COMBINAISON NON LINERAIRES DES VARIABLES INITIALES ###
from  sklearn.preprocessing import PolynomialFeatures
# Transformations des caracteristiques
poly = PolynomialFeatures(2,interaction_only=True)
caracteristics_poly = poly.fit_transform(caracteristics,)
# Combinaison des ancienes caracteristiques avec ACP
caracteristics_poly = np.append(caracteristics_pca, caracteristics_poly, axis=1)
# Lancement des classifieurs et affichage des ameliorations
clfs_stat_comb = run_classifiers(clfs,caracteristics_poly, target)
amelioration(clfs_stat_pca,clfs_stat_comb)

'''
###########################################################
#                   PARTIE 2                              #
###########################################################

# TRAITEMENT DES DONNEES MANQUANTES

# Transform strings to integers
X_cat = np.copy(values[:, INDICES_DISCRET])
X_new_cat = X_cat
for col_id in range(len(INDICES_DISCRET)):
    unique_val, val_idx = np.unique(X_cat[:, col_id], return_inverse=True)
    X_new_cat[:, col_id] = val_idx

    
# Select columns that contain missing values
replaceCols = np.any(values[:, INDICES_DISCRET]=="?", axis = 0)
catCols = X_new_cat[:, replaceCols]

# Replace missing values within categorial columns
from sklearn.preprocessing import Imputer
inpCat = Imputer(missing_values=0, strategy="most_frequent", axis=0)
catCols = inpCat.fit_transform(catCols)

# Replace missing values within numerical columns
numCols = values[:, INDICES_CONTINUS]
numCols[numCols=="?"]=np.nan
numCols=numCols.astype(np.float)
inpCont = Imputer(missing_values='NaN', strategy="mean", axis=0)
inpCont.fit(numCols)
numCols = inpCont.transform(numCols)

# Agregate all categorial columns back together (columns we just replaced missing values + columns that didn't contain any)
noUnknownCat = X_new_cat[:, ~np.any(values[:, INDICES_DISCRET]=="?", axis = 0)]
catCols = np.hstack((catCols, noUnknownCat))



# TRAITEMENT DE VARIABLES CATEGORIELLES
# HotEncoder for categorial values
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
catCols = enc.fit_transform(catCols).toarray()


# CONSTRUCTION DE VOTRE JEU DE DONNEES
# Reagregate all columns together
replacedvalues = np.hstack((catCols, numCols))

# Update target column to integrate lines with replaced missing values
target = values[:, 15]
target[target == '+'] = 1
target[target == '-'] = 0

# In case we need to re-aggregate it with all attributes columns
#target = np.transpose(target[np.newaxis]) # Transform target to a 2d array




# SELECTION DE VARIABLE

  
from sklearn.feature_selection import SelectKBest, mutual_info_classif
selection = SelectKBest(k=7, score_func=mutual_info_classif)
selected = selection.fit_transform(replacedvalues, np.asarray(target, dtype=np.int))

# Print boxplot
plt.bar(range(46), selection.scores_)
plt.show()
# faire des test




