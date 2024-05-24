import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use("ggplot") 

import warnings
warnings.filterwarnings('ignore')


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC 

from ucimlrepo import fetch_ucirepo 

# Fetch-ujemo dataset 
wine_quality = fetch_ucirepo(id=186) 
  
# data (pandas dataframes) 
X = wine_quality.data.features 
y = wine_quality.data.targets 
   
# Stampamo metadata 
print(wine_quality.metadata) 
  
# Stampamo informacije o promenljivim 
print(wine_quality.variables) 

df = wine_quality.data.features 
df['quality'] = wine_quality.data.targets

#Stampanje jedinstvenih vrednosti kvaliteta u datasetu i oblika dataseta
print("The Value Quality ",df["quality"].unique())
print("Oblik DataSet-a ", df.shape )

#Ispisujemo kratki pregled df-a i broj nedostajecih vrednosti za svaku kolonu
df.info()
print(df.isna().sum())


#Feature-i koji nemaju veliki uticaj na kvalitet
plt.figure(figsize=(20,7))
sns.lineplot(data=df, x="quality",y="volatile_acidity",label="Volatile Acidity")
sns.lineplot(data=df, x="quality",y="citric_acid",label="Citric Acid")
sns.lineplot(data=df, x="quality",y="chlorides",label="chlorides")
sns.lineplot(data=df, x="quality",y="pH",label="PH")
sns.lineplot(data=df, x="quality",y="sulphates",label="Sulphates")
plt.ylabel("Kvantitet")
plt.title("Manji uticaj na kvalitet")
plt.legend()
plt.savefig("plot1.png",format="png")

#Uticaj alkohola na kvalitet vina
plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="quality",y="alcohol")
plt.savefig("plot2.png",format="png")

#Uticaj sumpor dioksida na kvalitet vina
plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="quality",y="total_sulfur_dioxide",color="b")
plt.savefig("plot3.png",format="png")

#Uticaj slobodnog sumpor dioksida na kvalitet vina
plt.figure(figsize=(15,7))
sns.lineplot(data=df, x="quality",y="free_sulfur_dioxide",color="g")
plt.savefig("plot4.png",format="png")

plt.show()


#Podela na test i trening - 20:80 %
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("X Train : ", X_train.shape)
print("X Test  : ", X_test.shape)
print("Y Train : ", y_train.shape)
print("Y Test  : ", y_test.shape)

#Konverzija ciljnih promenljivih u jednodimenzionalne nizove (vektore)
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

accuracies = []

#Koriscenje MLP Classifier-a za treniranje
mlp = MLPClassifier(hidden_layer_sizes=(500,100),max_iter=300)
mlp.fit(X_train,y_train)
y_mlp_pred = mlp.predict(X_test)

print("Ocena za X-train sa Y-train je : ", mlp.score(X_train,y_train))
print("Ocena za X-test sa Y-test je : ", mlp.score(X_test,y_test))
print("Evaluacija MLP-a : Accuracy score " , accuracy_score(y_test,y_mlp_pred))
accuracies.append(accuracy_score(y_test,y_mlp_pred))

#Koriscenje Logicke Regresije (svodjenje na klasifikaciju) za treniranje
Lo_model=LogisticRegression(solver='liblinear')

Lo_model.fit(X_train,y_train)

print("Ocena za X-train sa Y-train je : ", Lo_model.score(X_train,y_train))
print("Ocena za X-test sa Y-test je : ", Lo_model.score(X_test,y_test))

y_pred_Lo=Lo_model.predict(X_test)
print("Evaluacija Logicke Regresije : Accuracy score " , accuracy_score(y_test,y_pred_Lo))
accuracies.append(accuracy_score(y_test,y_pred_Lo))


# Koriscenje Decision Tree Classifiera za treniranje
Tree_model = DecisionTreeClassifier(random_state=42)

# Definisanje grida hiperparametara radi pronalazenja optimalnih vrednosti hiperparametara
param_grid = {
    'max_depth': [20, 25, 30, 35, 40],
    'min_samples_split': [2, 3, 4, 5],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': [None, 'sqrt', 'log2'],
    'class_weight': [None, 'balanced']
}

#Pretraga najboljih parametara
grid_search = GridSearchCV(Tree_model, param_grid, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

#Overrideujemo Tree model novim objektom sa optimalnim hiperparametrima
Tree_model = DecisionTreeClassifier(**grid_search.best_params_)
Tree_model.fit(X_train,y_train)

y_pred_tree =Tree_model.predict(X_test)

print("Ocena za X-train sa Y-train je : ", Tree_model.score(X_train,y_train))
print("Ocena za X-test sa Y-test je : ", Tree_model.score(X_test,y_test))
print("Evaluacija Decision Tree : Accuracy score " , accuracy_score(y_test,y_pred_tree))

accuracies.append(accuracy_score(y_test,y_pred_tree))

# Koriscenje SVC za treniranje
svc_model=SVC(C=50,kernel="rbf")

svc_model.fit(X_train,y_train)

y_pred_svc =svc_model.predict(X_test)

print("Ocena za X-train sa Y-train je : ", svc_model.score(X_train,y_train))
print("Ocena za X-test sa Y-test je : ", svc_model.score(X_test,y_test))
print("Evaluacija SVC-a : Accuracy score " , accuracy_score(y_test,y_pred_svc))

accuracies.append(accuracy_score(y_test,y_pred_svc))


#Koriscenje KNeighbors Classifier-a za treniranje
K_model = KNeighborsClassifier(n_neighbors = 5)
K_model.fit(X_train, y_train)

y_pred_k = K_model.predict(X_test)

print("Ocena za X-train sa Y-train je : ", K_model.score(X_train,y_train))
print("Ocena za X-test sa Y-test je : ", K_model.score(X_test,y_test))
print("Evaluacija K Neighbors-a : Accuracy score " , accuracy_score(y_test,y_pred_k))

accuracies.append(accuracy_score(y_test,y_pred_k))


x = np.array(accuracies)
y = np.sort(x)

plt.title("bar")
plt.bar(x, y, color="red")

plt.show()