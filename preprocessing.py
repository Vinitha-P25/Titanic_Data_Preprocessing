import pandas as pd

# ----------------------------------------
# Step 1 : Load Dataset
# ----------------------------------------

df = pd.read_csv("data/train.csv")

print("\n===== TITANIC DATA PREPROCESSING =====\n")

# ----------------------------------------
# Step 2 : Display Dataset
# ----------------------------------------

print("First 5 Rows")
print(df.head())

print("\nLast 5 Rows")
print(df.tail())

# ----------------------------------------
# Step 3 : Dataset Information
# ----------------------------------------

print("\nShape of Dataset")
print(df.shape)

print("\nColumn Names")
print(df.columns)

print("\nData Types")
print(df.dtypes)

print("\nDataset Information")
print(df.info())

# ----------------------------------------
# Step 4 : Statistical Summary
# ----------------------------------------

print("\nStatistical Summary")
print(df.describe())

# ----------------------------------------
# Step 5 : Missing Values
# ----------------------------------------

print("\nMissing Values")
print(df.isnull().sum())

# ----------------------------------------
# Step 6 : Fill Missing Values
# ----------------------------------------

# Age -> Median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Embarked -> Mode
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin has many missing values, remove it
df.drop("Cabin", axis=1, inplace=True)

print("\nMissing Values After Cleaning")
print(df.isnull().sum())

# ----------------------------------------
# Step 7 : Duplicate Records
# ----------------------------------------

print("\nDuplicate Rows")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)

print("Duplicates Removed")

# ----------------------------------------
# Step 8 : Statistics
# ----------------------------------------

print("\nAverage Age")
print(df["Age"].mean())

print("\nMedian Age")
print(df["Age"].median())

print("\nMode of Embarked")
print(df["Embarked"].mode())

print("\nMaximum Fare")
print(df["Fare"].max())

print("\nMinimum Fare")
print(df["Fare"].min())

print("\nAge Count")
print(df["Age"].count())

print("\nStandard Deviation of Age")
print(df["Age"].std())

print("\nVariance of Age")
print(df["Age"].var())

# ----------------------------------------
# Step 9 : Unique Values
# ----------------------------------------

print("\nUnique Gender Values")
print(df["Sex"].unique())

print("\nUnique Passenger Classes")
print(df["Pclass"].unique())

# ----------------------------------------
# Step 10 : Value Counts
# ----------------------------------------

print("\nGender Count")
print(df["Sex"].value_counts())

print("\nSurvival Count")
print(df["Survived"].value_counts())

# ----------------------------------------
# Step 11 : Filtering
# ----------------------------------------

print("\nPassengers Age > 30")
print(df[df["Age"] > 30])

print("\nFemale Passengers")
print(df[df["Sex"] == "female"])

# ----------------------------------------
# Step 12 : Sorting
# ----------------------------------------

print("\nHighest Fare")
print(df.sort_values(by="Fare", ascending=False).head())

print("\nYoungest Passengers")
print(df.sort_values(by="Age").head())

# ----------------------------------------
# Step 13 : Feature Engineering
# ----------------------------------------

df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

print("\nFamilySize Column Added")

# ----------------------------------------
# Step 14 : Rename Column
# ----------------------------------------

df.rename(columns={"Pclass": "PassengerClass"}, inplace=True)

print("\nColumn Renamed")

# ----------------------------------------
# Step 15 : Group By
# ----------------------------------------

print("\nAverage Fare by Passenger Class")
print(df.groupby("PassengerClass")["Fare"].mean())

print("\nSurvival Rate by Gender")
print(df.groupby("Sex")["Survived"].mean())

# ----------------------------------------
# Step 16 : Correlation
# ----------------------------------------

print("\nCorrelation Matrix")
print(df.corr(numeric_only=True))

# ----------------------------------------
# Step 17 : Save Cleaned Dataset
# ----------------------------------------

df.to_csv("output/Titanic_Cleaned.csv", index=False)

print("\nCleaned Dataset Saved Successfully!")