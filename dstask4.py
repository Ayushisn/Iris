import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

file_path = 'C:\\Users\\Atulya Kumar\\Downloads\\Iris.csv'  
data = pd.read_csv(file_path)

print(data.head())

print(data.info())

print(data.describe())

print(data['species'].unique())

X = data.drop('species', axis=1)  
y = data['species']  

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

print("Random Forest Classifier:")
print("Accuracy:", accuracy_score(y_test, rf_predictions))
print(classification_report(y_test, rf_predictions, target_names=label_encoder.classes_))

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
knn_predictions = knn_model.predict(X_test)

print("k-Nearest Neighbors Classifier:")
print("Accuracy:", accuracy_score(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions, target_names=label_encoder.classes_))

svc_model = SVC(kernel='linear', random_state=42)
svc_model.fit(X_train, y_train)
svc_predictions = svc_model.predict(X_test)

print("Support Vector Classifier:")
print("Accuracy:", accuracy_score(y_test, svc_predictions))
print(classification_report(y_test, svc_predictions, target_names=label_encoder.classes_))
