import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score 
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# This is the class for usage of Decision Tree classifier.
class DecisionTree_Classifier:

    # Initially for the init method, we have to pass the required file names to generate the train and test data for the classifier
    def __init__(self,train_input_features, train_target_labels, test_input_features, test_target_labels,maximum_depth):
        self.train_input_features = pd.read_csv(train_input_features)
        self.train_target_labels = pd.read_csv(train_target_labels)
        self.test_input_features = pd.read_csv(test_input_features)
        self.test_target_labels = pd.read_csv(test_target_labels)
        self.maximum_depth = maximum_depth
    
    # This method generates the model of Decision Tree classifer
    def build_DecisionTreeClass_Model(self):
        self.DecisionTree_Model = DecisionTreeClassifier(max_depth = self.maximum_depth)
        self.DecisionTree_Model = self.DecisionTree_Model.fit(self.train_input_features,self.train_target_labels)

    # This method generates the confusion matrix and also prints it.
    def generate_ConfusionMatrix(self):
        self.prediction_y = self.DecisionTree_Model.predict(self.test_input_features)
        confusionMatrix = confusion_matrix(self.test_target_labels,self.prediction_y)
        print(confusionMatrix)

    # This method calculates all the performance metrics required to assess the classifier performance
    def generate_Performace_Metrics(self):
        DecisionTree_accuracy = self.DecisionTree_Model.score(self.test_input_features,self.test_target_labels)*100
        DecisionTree_precision = precision_score(self.test_target_labels,self.prediction_y)*100
        DecisionTree_recall = recall_score(self.test_target_labels, self.prediction_y)*100
        DecisionTree_f1=f1_score(self.test_target_labels, self.prediction_y)*100 

        # Printing out the Accuracy , Precision , Recall and F1 Score
        print("Decision Tree Accuracy : {0:.5f}%".format(DecisionTree_accuracy))
        print("Precision : {0:.5f}%".format(DecisionTree_precision))
        print("Recall : {0:.5f}%".format(DecisionTree_recall))
        print("F1 Score : {0:.5f}%".format(DecisionTree_f1))

# The program execution starts from here and creates a new object of the DecisionTree_Classifier class and performs the required method on the object
if __name__ == '__main__':
    decisionTreeObj = DecisionTree_Classifier('train_input_features.csv','train_target_labels.csv','test_input_features.csv','test_target_labels.csv',10)
    decisionTreeObj.build_DecisionTreeClass_Model()
    decisionTreeObj.generate_ConfusionMatrix()
    decisionTreeObj.generate_Performace_Metrics()




