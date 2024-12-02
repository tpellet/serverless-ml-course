
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import seaborn as sns

iris_df = pd.read_csv("src/01-module/assets/dataset.csv")
iris_df.sample(10)

features = iris_df[["sepal_length", "sepal_width", "petal_length", "petal_width"]]
labels = iris_df[["variety"]]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(features, labels, test_size=0.2)
model = KNeighborsClassifier(n_neighbors=2)
model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)

from sklearn.metrics import classification_report

metrics = classification_report(y_test, y_pred, output_dict=True)
from sklearn.metrics import confusion_matrix

results = confusion_matrix(y_test, y_pred)
from matplotlib import pyplot

df_cm = pd.DataFrame(results, ['True Setosa', 'True Versicolor', 'True Virginica'],
                     ['Pred Setosa', 'Pred Versicolor', 'Pred Virginica'])

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def iris(sepal_length, sepal_width, petal_length, petal_width):
    input_list = []
    input_list.append(sepal_length)
    input_list.append(sepal_width)
    input_list.append(petal_length)
    input_list.append(petal_width)
    # 'res' is a list of predictions returned as the label.
    res = model.predict(np.asarray(input_list).reshape(1, -1)) 
    # We add '[0]' to the result of the transformed 'res', because 'res' is a list, and we only want 
    # the first element.
    flower_name = str(res[0])
    return flower_name

st.title("Iris Flower Predictive Analytics")
st.write("Experiment with sepal/petal lengths/widths to predict which flower it is.")

sepal_length = st.number_input("sepal length (cm)", value=1.0)
sepal_width = st.number_input("sepal width (cm)", value=1.0)
petal_length = st.number_input("petal length (cm)", value=1.0)
petal_width = st.number_input("petal width (cm)", value=1.0)

if st.button("Predict"):
    flower_name = iris(sepal_length, sepal_width, petal_length, petal_width)
    img = imread(f"src/01-module/assets/{flower_name}.png")
    plt.imshow(img)
    plt.axis("off")
    st.pyplot(plt)
