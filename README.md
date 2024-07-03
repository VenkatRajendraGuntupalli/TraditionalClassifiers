
# Traditional Machine Learning Classification
This project implements three traditional classifiers - Decision Tree Classifier, K-Nearest Neighbors (KNN) Classifier, and Random Forest Classifier. These classifiers are applied to a heart disease dataset to predict the presence or absence of heart disease.

#### Author: Mr.Guntupalli


### Dataset
The dataset used in this project is heart.csv, which contains various features related to heart health and a target variable indicating the presence of heart disease.

### Data Division
The Data_Divider.py script is used to divide the dataset into training and testing sets. It splits the data into input features and target labels and saves them into separate CSV files: train_input_features.csv, train_target_labels.csv, test_input_features.csv, and test_target_labels.csv.

### Decision Tree Classifier
The DecisionTreeClassifier.py script implements the Decision Tree Classifier. It takes the training and testing data files as inputs and builds a decision tree model with a specified maximum depth. The script then generates a confusion matrix and performance metrics including accuracy, precision, recall, and F1 score for the model.

### K-Nearest Neighbors (KNN) Classifier
The Knn_Classifier.py script implements the K-Nearest Neighbors Classifier. Similar to the Decision Tree Classifier, it takes the training and testing data files as inputs and builds a KNN model with a specified number of neighbors. The script generates a confusion matrix and performance metrics for the KNN model.

### Random Forest Classifier
The RandomForest_Classifier.py script implements the Random Forest Classifier. It takes the training and testing data files, maximum depth, and number of estimators as inputs. The script builds a random forest model with the specified parameters and generates a confusion matrix and performance metrics for the model.

### Usage
To use this project, follow these steps:

- Ensure that the required dependencies (scikit-learn, pandas) are installed.
- Run the Data_Divider.py script to split the dataset into training and testing sets.
- Run any of the classifier scripts (DecisionTreeClassifier.py, Knn_Classifier.py, RandomForest_Classifier.py) to build and evaluate the corresponding classifier.
Note: Modify the input file names and parameters as needed for your specific use case.

### Results
The generated confusion matrices and performance metrics will be displayed in the console when running each classifier script. These metrics provide insights into the performance of the classifiers on the heart disease dataset.

Feel free to explore and modify the code to experiment with different classifier parameters or datasets.

### Conclusion
- This project demonstrates the application of traditional classifiers (Decision Tree, KNN, and Random Forest) for predicting heart disease. 
- The README.md file provides an overview of the project structure, dataset, classifiers used, and instructions for usage.

## Run Locally

Clone the project

```bash
  git clone https://github.com/VenkatRajendraGuntupalli/TraditionalClassifiers.git
```
