import pandas as pd

# Load df & check for duplicate rows & null values
df = pd.read_csv("energydata.csv")
print(df.duplicated().sum())
print(df.isna().sum())

# Convert date feature into separate numerical columns
df['date'] = pd.to_datetime(df['date'])
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df['day'] = df['date'].dt.day
df['hour'] = df['date'].dt.hour
df['minute'] = df['date'].dt.minute
df = df.drop(['date'], axis=1)
print(df[['year', 'month', 'day', 'hour', 'minute']])