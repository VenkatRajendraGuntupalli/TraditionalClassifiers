import pandas as pd
from sklearn.model_selection import train_test_split

# Load CSV file into pandas DataFrame
req_dataFrame = pd.read_csv('heart.csv')

# Split training data into input features and target labels
X = req_dataFrame.iloc[:,:-1]
y = req_dataFrame['HeartDisease']

# Split data into training and testing datasets
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

# Save training and testing datasets to CSV files
train_x.to_csv('train_input_features.csv', index=False)
train_y.to_csv('train_target_labels.csv', index=False)
test_x.to_csv('test_input_features.csv', index=False)
test_y.to_csv('test_target_labels.csv', index=False)
