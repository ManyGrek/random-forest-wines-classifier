# Izael Manuel Rascón Durán A01562240

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# Load the wines dataset
wines = datasets.load_wine()

X = wines.data
y = wines.target

# Split dataset for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Initialize model
rfc = RandomForestClassifier()

# Train model
rfc.fit(X_train, y_train)

# Make predictions
y_predicted = rfc.predict(X_test)

# Evaluations
print('Evaluation scores')
print('Accuracy:', accuracy_score(y_test, y_predicted))
print('Confusion matrix:\n', confusion_matrix(y_test, y_predicted))
print('F1 score:', f1_score(y_test, y_predicted, average='macro'))

