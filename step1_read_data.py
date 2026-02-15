import pandas as pd
import matplotlib.pyplot as plt


# read the csv file
df = pd.read_csv("paddy_data.csv")

print("Dataset Loaded Successfully\n")

print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)

print("\nBasic Statistics:")
print(df.describe())

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

#to separate features and targets
x = df[["Temp_Day" , "Temp_Night" , "Humidity" , "Rainfall", "Temp_Drop", "Avg_Temp"]]
y = df["Disease"]

#split data (80% training and 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#create model
model = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

#Train the model
model.fit(x_train, y_train)

#Predict on the test set
y_pred = model.predict(x_test)

#check accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

plt.figure()
plt.scatter(df["Humidity"], df["Disease"], alpha=0.6)
plt.xlabel("Humidity (%)")
plt.ylabel("Disease (0 = No, 1 = Yes)")
plt.title("Effect of Humidity on Paddy Disease Occurrence")
plt.grid(True)
plt.show()

plt.figure()
df["Disease"].value_counts().plot(kind="bar")
plt.xlabel("Disease (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.title("Disease Distribution in Dataset")
plt.show()



print("\n ---Paddy Disease Risk Prediction---")

#take user input
temp_day = float(input("Enter Day Temperature: "))
temp_night = float(input("Enter Night Temperature: "))
humidity = float(input("Enter Humidity(%): "))
rainfall = float(input("Enter Rainfall(mm): "))
temp_drop = temp_day - temp_night


#create DataFrame for prediction
new_data = pd.DataFrame({
    "Temp_Day": [temp_day],
    "Temp_Night": [temp_night],
    "Humidity": [humidity],
    "Rainfall": [rainfall],
    "Temp_Drop": [temp_drop]
})

#predict probability
probability = model.predict_proba(new_data)
risk_percentage = probability[0][1] * 100

print(f"\nDisease Risk: {risk_percentage:.2f}%")

#risk level classification
if risk_percentage > 60:
    print("Risk Level: HIGH")
elif risk_percentage > 30:
    print("Risk Level: MEDIUM")
else:
    print("Risk Level: LOW")