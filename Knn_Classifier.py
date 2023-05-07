import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# This is the class for usage of KNN classifier.
class KNN_Classifier:
    # Initially for the init method, we have to pass the required file names to generate the train and test data for the classifier
    def __init__(self,train_input_features, train_target_labels, test_input_features, test_target_labels,n_neighbours):
        self.train_input_features = pd.read_csv(train_input_features)
        self.train_target_labels = pd.read_csv(train_target_labels)
        self.test_input_features = pd.read_csv(test_input_features)
        self.test_target_labels = pd.read_csv(test_target_labels)
        self.n_neighbours = n_neighbours

    # This method generates the model of Decision Tree classifer
    def build_KNNClassifier(self):
        self.KNN_model = KNeighborsClassifier(n_neighbors = self.n_neighbours)
        self.KNN_model = self.KNN_model.fit(self.train_input_features, self.train_target_labels)

    # This method generates the confusion matrix and also prints it.
    def generate_ConfusionMatrix(self):
        self.prediction_y = self.KNN_model.predict(self.test_input_features)
        confusionMatrix = confusion_matrix(self.test_target_labels,self.prediction_y)
        print(confusionMatrix)

    # This method calculates all the performance metrics required to assess the classifier performance
    def generate_Performace_Metrics(self):
        KNN_accuracy = self.KNN_model.score(self.test_input_features, self.test_target_labels)*100
        KNN_precision = precision_score(self.test_target_labels, self.prediction_y)*100
        KNN_recall = recall_score(self.test_target_labels, self.prediction_y)*100
        KNN_f1=f1_score(self.test_target_labels, self.prediction_y)*100

        # Printing out the Accuracy , Precision , Recall and F1 Score
        print("KNN Accuracy : {0:.5f}%".format(KNN_accuracy))
        print("KNN F1 Score : {0:.5f}%".format(KNN_f1))
        print("KNN Recall : {0:.5f}%".format(KNN_recall))
        print("KNN Precision : {0:.5f}%".format(KNN_precision))

# The program execution starts from here and creates a new object of the KNN_Classifier class and performs the required method on the object
if __name__ == '__main__':
    knnModelObj = KNN_Classifier('train_input_features.csv','train_target_labels.csv','test_input_features.csv','test_target_labels.csv',10)
    knnModelObj.build_KNNClassifier()
    knnModelObj.generate_ConfusionMatrix()
    knnModelObj.generate_Performace_Metrics()
