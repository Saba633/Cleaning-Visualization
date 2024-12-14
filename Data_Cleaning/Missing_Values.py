# --------------------------------------------- Dataset -----------------------------------------

import pandas as pd

data  = pd.read_csv(r"D:\Code\Python\ML\dataasets\medical_conditions_dataset.csv")
df = pd.DataFrame(data=data)

print(df)
# --------------------------------------------- Fillna() method-----------------------------------------

print("\nFillna method\n")

data  = pd.read_csv(r"D:\Code\Python\ML\dataasets\medical_conditions_dataset.csv")
df = pd.DataFrame(data=data)

# null_val_cols =  df.isnull().any()

df['age'] = df['age'].fillna(df['age'].mean()).astype(int)
df[['bmi', 'glucose_levels']] = df[['bmi', 'glucose_levels']].fillna(df[['bmi', 'glucose_levels']].median()).round(2)
df['blood_pressure'] = df['blood_pressure'].fillna(df['blood_pressure'].median()).round(4)

print(df)

# --------------------------------------------- SimpleImputermethod-----------------------------------------

from sklearn.impute import SimpleImputer 

print("\nSimple Imputer\n")
data  = pd.read_csv(r"D:\Code\Python\ML\dataasets\medical_conditions_dataset.csv")
df = pd.DataFrame(data=data)

imputer = SimpleImputer(strategy='mean')
df[['age', 'bmi', 'blood_pressure', 'glucose_levels']] = imputer.fit_transform(df[['age', 'bmi', 'blood_pressure', 'glucose_levels']])
df['age'] = df['age'].astype(int)

# print(df)

# --------------------------------------------- Interpoltion -----------------------------------------

df = df.infer_objects()

df_numeric = df.select_dtypes(include=['float64', 'int64']) #select numerics only

interpolating_numerics = df_numeric.interpolate(method="linear") #interploating columns
interpolating_numerics = interpolating_numerics.ffill().bfill().round(3) #filling values from front and back

interpolating_numerics['age'] = interpolating_numerics['age'].astype(int) #converting age to int

print(f"\nFilling in missing values using Interploating method \n\n{interpolating_numerics}")
