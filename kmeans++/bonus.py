from sklearn import datasets
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def main():
    iris = datasets.load_iris()
    x = range(1,11)
    y = []
    for i in x:
        model = KMeans(n_clusters=i, init='k-means++', random_state=0)
        model.fit(iris.data)
        y.append(model.inertia_)

    plt.plot(x,y)
    plt.title('Elbow Method for selection of optimal "K" clusters')
    plt.xlabel('K')
    plt.ylabel('Average Dispersion')
    plt.xticks(x)
    plt.annotate('Elbow Point', xy = (3,y[2]), xytext =(3,y[2]+150), arrowprops = dict(facecolor ='green', shrink = 0.05))
    plt.savefig('elbow.png')
    plt.close()

if __name__ == "__main__":
    main()
