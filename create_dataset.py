import pandas as pd
import numpy as np

np.random.seed(42)

rows = 300

temp_day = np.random.randint(28, 38, rows)
temp_night = np.random.randint(20, 28, rows)
humidity = np.random.randint(60, 98, rows)
rainfall = np.random.randint(0, 30, rows)
avg_temp = (temp_day + temp_night) / 2

temp_drop = temp_day - temp_night

disease = []

for i in range(rows):
    risk_score = 0
    
    if humidity[i] > 85:
        risk_score += 1
    if rainfall[i] > 15:
        risk_score += 1
    if temp_drop[i] > 8:
        risk_score += 1
    
    if risk_score >= 2:
        disease.append(np.random.choice([1, 1, 1, 1, 0]))
    else:
        disease.append(np.random.choice([0, 0, 0, 0, 1]))

df = pd.DataFrame({
    "Temp_Day": temp_day,
    "Temp_Night": temp_night,
    "Humidity": humidity,
    "Rainfall": rainfall,
    "Temp_Drop": temp_drop,
    "Avg_Temp": avg_temp,
    "Disease": disease
})

df.to_csv("paddy_data.csv", index=False)

print("Realistic dataset created successfully!")
print(df.head())
