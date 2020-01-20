from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import argparse
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def train():
    data = pickle.loads(open("./output/embeddings.pickle", "rb").read())
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    #recognizer = SVC(C=1.0, kernel="linear", probability=True)
    recognizer = GradientBoostingClassifier(n_estimators=500, random_state=0)
    #recognizer = KNeighborsClassifier(n_neighbors=5, algorithm='ball_tree')
    recognizer.fit(data["embeddings"], labels)

    # write the actual face recognition model to disk
    f = open("./output/recognizer.pickle", "wb")
    f.write(pickle.dumps(recognizer))
    f.close()
    
    # write the label encoder to disk
    f = open("./output/le.pickle", "wb")
    f.write(pickle.dumps(le))
    f.close()

def plot():
    data = pickle.loads(open("./output/embeddings.pickle", "rb").read())
    le = LabelEncoder()
    labels = le.fit_transform(data["names"])

    df = TSNE(n_components=2).fit_transform(data["embeddings"])
    
    ax = sns.scatterplot(x=df[:,0], y=df[:,1], hue=data["names"])
    plt.show()

    

if __name__ == "__main__":
    #plot()
    train()
