import pickle
import pandas as pd

# Load your dataset
data = pd.read_csv("traffic.csv")

# Convert 'DateTime' column to datetime format
data["DateTime"] = pd.to_datetime(data["DateTime"])

# Create a binary target variable indicating whether there is traffic (1) or not (0)
data["Traffic"] = (data["Vehicles"] > 10).astype(int)

# Extract features using .copy() to ensure a copy of the DataFrame is created
features = data[["Junction", "DateTime"]].copy()

# Use .loc to set values for 'Hour' and 'DayOfWeek' columns
features.loc[:, "Hour"] = data["DateTime"].dt.hour.values
features.loc[:, "DayOfWeek"] = data["DateTime"].dt.dayofweek.values

# Drop the original 'DateTime' column
features.drop("DateTime", axis=1, inplace=True)


# Extract target variable
target = data["Traffic"]

# Drop rows with missing values
data.dropna(inplace=True)

# Or, impute missing values with the mean
data.fillna(data.mean(), inplace=True)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# Example: Create a feature indicating whether it's a weekday or weekend
features["IsWeekend"] = features["DayOfWeek"].apply(lambda x: 1 if x >= 5 else 0)

# One-hot encoding for categorical feature 'Junction'
features_encoded = pd.get_dummies(features, columns=["Junction"], drop_first=True)

from sklearn.ensemble import IsolationForest

outlier_detector = IsolationForest(contamination=0.1)
outliers = outlier_detector.fit_predict(features)
cleaned_data = data[outliers == 1]

from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)

# Now, use SMOTE to oversample the data
X_resampled, y_resampled = smote.fit_resample(features, target)

from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

from sklearn.ensemble import RandomForestClassifier  # type: ignore
from sklearn.metrics import accuracy_score, classification_report  # type: ignore

# Train a RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_train)


pickle.dump(classifier, open("model.pkl", "wb"))

# Make predictions on the test set
# predictions = classifier.predict(X_test)

# Evaluate the model
# accuracy = accuracy_score(y_test, predictions)
# print(f"Accuracy: {accuracy}")
# print("Classification Report:")
# print(classification_report(y_test, predictions))
