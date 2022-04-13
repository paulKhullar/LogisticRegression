import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split 
from sklearn.metrics import balanced_accuracy_score
from sklearn import preprocessing
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sn
import matplotlib.pyplot as plt  

#column in the dataset that our model will predict
result_column = [29]
#features of the dataset used to make predictions
data_column = [1,2,3,4,5,6,7,8,9,10,11,12]
file_name = 'drug_consumption.data'

def train_model(x_train,y_train) :
    model = LogisticRegression(penalty='none',class_weight=None, max_iter=10000, verbose=1,tol=.000001)
#    model = LogisticRegression(penalty='none',class_weight='balanced', max_iter=10000, verbose=1,tol=.000001)
    model.fit(x_train,y_train.values.ravel())
    return model


def process_data(file_name) :
    scaler = preprocessing.MinMaxScaler()
    training = pd.read_csv(file_name, usecols=data_column, index_col=None)
    results = pd.read_csv(file_name, usecols=result_column, skiprows=None, index_col=None)

    #normalise the data so that it is more usable for our model
    scaler.fit(training)
    training = scaler.transform(training)

    #split the dataset up, by train, test, and validation sets
    x_train, x_testing, y_train, y_testing = train_test_split(training, results, test_size=0.4, shuffle=False,
    random_state=2)
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size=0.4,
    shuffle=False)
    return x_train, y_train, x_testing, y_testing, x_validation, y_validation


def evaluate_model(model, x_test,y_test) :
    print(model.score(x_test, y_test))
    y_pred = model.predict(x_test)
    print(balanced_accuracy_score(y_test,y_pred))
    print(precision_score(y_test,y_pred, average='macro'))
    print(recall_score(y_test,y_pred, average='macro'))
    print(cross_val_score(model,x_test,y_test,cv=2))

def confusion_matrices(model, x_test, y_test) :
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_test,y_pred)
    ax= plt.subplot()
    sn.heatmap(cm, annot=True, fmt='g', ax=ax,cmap='Blues_r')
    ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels')
    plt.show()


x_train,y_train, x_test, y_test, x_validation, y_validation = process_data(file_name)
model = train_model(x_train, y_train )
evaluate_model(model, x_test,y_test)
#confusion_matrices(model, x_test,y_test)
