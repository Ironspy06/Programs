from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris 
from sklearn import metrics
from sklearn.model_selection import train_test_split as split
dataset=load_iris()

X=dataset.data

y=dataset.target

X_train,X_test,y_train,y_test=split(X,y, test_size=0.2, random_state=1) 
gnb=GaussianNB()

classifier=gnb.fit(X_train,y_train) 
y_pred=classifier.predict(X_test)

print ('Accuracy matrices',metrics.classification_report(y_test,y_pred))
print ('Accuracy of the classifier is', metrics.accuracy_score (y_test,y_pred))

print("Confusion matrices")

print(metrics.confusion_matrix(y_test, y_pred))
