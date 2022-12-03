import pylab as pl
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA as RandomizedPCA
from sklearn.svm import SVC
import time 
from sklearn.metrics import accuracy_score


lfw_people = fetch_lfw_people(min_faces_per_person=70,resize=0.4)
n_samples, h, w = lfw_people.images.shape
np.random.seed(42)

X = lfw_people.data
Y = lfw_people.target

n_feat = X.shape[1]
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Date set size:")
print("n sample:",n_samples)
print("n feat:",n_feat)
print("n_classes:",n_classes)


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
n_components = 50
s_time = time.time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
e_time = time.time()
all_time_procces = e_time - s_time
print("This procces took %i second "%(all_time_procces))

eigenfaces = pca.components_.reshape((n_components, h, w))
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

print("Fitting the classifier")
param_grid = {
         'C': [1e3, 5e3, 1e4, 5e4, 1e5],
          'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
          }
clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
clf = clf.fit(X_train_pca, y_train)
print("Best estimator found:")
print(clf.best_estimator_)


print("Predicting the people names on the testing set")
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))
print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)
def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())

prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]
plot_gallery(X_test, prediction_titles, h, w)
eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
plot_gallery(eigenfaces, eigenface_titles, h, w)
plt.show()
score = accuracy_score(y_test, y_pred)
print(classification_report(y_test, y_pred))
print(score)

