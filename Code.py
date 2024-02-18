import os
import numpy as np
import pickle
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# prepare Data

input_path = "D:\Data_Scientist\img_classification sklearn\clf-data"
categories = ["empty","not_empty"]

Data = []
Labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_path, category)):
        img_path = os.path.join(input_path, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        Data.append(img.flatten())
        Labels.append(category_idx)

Data = np.asarray(Data)
Labels = np.asarray(Labels)

# train / test data

x_train, x_test, y_train, y_test = train_test_split(Data, Labels, test_size=0.2, shuffle=True, stratify=Labels)

# train_classifiers

classifier = SVC()
parameters = [{"gamma": [0.1, 0.001, 0.0001],"C": [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, parameters)
grid_search.fit(x_train, y_train)

# test performance

best_estimater = grid_search.best_estimator_
y_prediction = best_estimater.predict(x_test)
score = accuracy_score(y_prediction, y_test)
print("{}% of samples were correctly".format(str(score*100)))

pickle.dump(best_estimater, open("./model.p", "wb"))