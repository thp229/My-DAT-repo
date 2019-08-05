import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import random as r

#randomly generated 1000 rows with a qualitative response
data = {"X1":[r.gauss(0,1) for i in range(500)]+[r.gauss(0,1) for i in range(500)],
        "X2":[r.uniform(0,5) for i in range(500)]+[r.uniform(5,10) for i in range(500)],
        "Y":[0 for i in range(500)] + [1 for i in range (500)]
}
X = pd.DataFrame.from_dict(data) 

def knn(k,target, data):
    prediction = None
    T = list(data.loc[target, :][:2])

    data["index"] = data.index.tolist()
    Remainder = data.loc[data.index != target, X.columns != "Y"]
    
    rankings = []
    for i in range(len(Remainder)):
        metric = 0
        for j in range(2):
            metric += (list(Remainder.iloc[i,:][:2])[j] - T[j])**2
        rankings.append(metric**(0.5))
    
    Remainder["ranking"] = rankings
    neighbors = Remainder.sort_values(by = ["ranking"]).index.tolist()[:k]

    def category(row):
        if row["index"] in neighbors:
            return 1
        elif row["index"] == target:
            return 2
        else:
            return 0
    data["category"] = data.apply(category, axis = 1)

    def get_prediction(neighbors):
        global prediction
        counter = 0
        for i in neighbors:
            if list(data.iloc[i ,: ])[2] == 1:
                counter += 1
        if counter/len(neighbors) > 0.5:
            prediction = 1
            return prediction
        else:
            prediction = 0 
            return prediction

    return data, get_prediction(neighbors)


##########################################################
target = 234

print("target row:", list(X.loc[target, :]))
neighbors_plot, prediction = knn(100,target,X)
if prediction == list(X.loc[target, :])[2]:
    print("The KNN was correct!")
else:
    print("The KNN was incorrect")

def viewPlots():
    fig , ax = plt.subplots(1,2, sharey = False, figsize = (14,5))
    sns.scatterplot(x  = "X1", y = "X2", hue = "category", data = neighbors_plot, ax = ax[0])
    sns.scatterplot(x  = "X1", y = "X2", hue = "Y", data = neighbors_plot, ax = ax[1])
    ax[0].title.set_text("Subset of 100 nearest neighbors to target value")
    ax[1].title.set_text("The decision boundary and target value"+"\n"+"The KNN predicted: "+str(prediction))
    plt.scatter(x=list(X.loc[target, :])[0], y=list(X.loc[target, :])[1], color='r')
    plt.show()

viewPlots()

