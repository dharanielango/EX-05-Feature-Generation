# EX-05-Feature-Generation


## AIM
To read the given data and perform Feature Generation process and save the data to a file. 

# Explanation
Feature Generation (also known as feature construction, feature extraction or feature engineering) is the process of transforming features into new features that better relate to the target.
 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature Generation techniques to all the feature of the data set
### STEP 4
Save the data to the file

# Program:
## Data.csv :
```
import pandas as pd
df=pd.read_csv("data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder,OneHotEncoder
import category_encoders as ce
be=ce.BinaryEncoder()
ohe=OneHotEncoder(sparse=False)
le=LabelEncoder()
oe=OrdinalEncoder()


df1["City"] = ohe.fit_transform(df1[["City"]])

temp=['Cold','Warm','Hot','Very Hot']
oe1=OrdinalEncoder(categories=[temp])
df1['Ord_1'] = oe1.fit_transform(df1[["Ord_1"]])

edu=['High School','Diploma','Bachelors','Masters','PhD']
oe2=OrdinalEncoder(categories=[edu])
df1['Ord_2']= oe2.fit_transform(df1[["Ord_2"]])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'City', 'Ord_1','Ord_2','Target'])
df5
~~~
## Encoding.csv :
~~~
import pandas as pd
df=pd.read_csv("Encoding Data.csv")
df

#feature generation
import category_encoders as ce
be=ce.BinaryEncoder()
ndf=be.fit_transform(df["bin_1"])
df["bin_1"] = be.fit_transform(df["bin_1"])
ndf

ndf2=be.fit_transform(df["bin_2"])
df["bin_2"] = be.fit_transform(df["bin_2"])
ndf2

df1=df.copy()
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
le=LabelEncoder()
oe=OrdinalEncoder()

df1["nom_0"] = oe.fit_transform(df1[["nom_0"]])
temp=['Cold','Warm','Hot']
oe2=OrdinalEncoder(categories=[temp])
df1['ord_2'] = oe2.fit_transform(df1[['ord_2']])

df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df0=pd.DataFrame(sc.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df0

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df2=pd.DataFrame(sc1.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df2

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df3=pd.DataFrame(sc2.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df3

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df4=pd.DataFrame(sc3.fit_transform(df1),columns=['id', 'bin_1', 'bin_2', 'nom_0','ord_2'])
df4
~~~

## Titanic.csv :
```

import pandas as pd
df=pd.read_csv("titanic_dataset.csv")
df

#removing unwanted data
df.drop("Name",axis=1,inplace=True)
df.drop("Ticket",axis=1,inplace=True)
df.drop("Cabin",axis=1,inplace=True)

#data cleaning
df.isnull().sum()

df["Age"]=df["Age"].fillna(df["Age"].median())
df["Embarked"]=df["Embarked"].fillna(df["Embarked"].mode()[0])

df.isnull().sum()

df

#feature encoding
from category_encoders import BinaryEncoder
be=BinaryEncoder()
df["Sex"]=be.fit_transform(df[["Sex"]])
ndf=be.fit_transform(df["Sex"])
ndf

df1=df.copy()
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
embark=['S','C','Q']
e1=OrdinalEncoder(categories=[embark])
df1['Embarked'] = e1.fit_transform(df[['Embarked']])
df1

#feature scaling
from sklearn.preprocessing import MinMaxScaler
sc=MinMaxScaler()
df2=pd.DataFrame(sc.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df2

from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
df3=pd.DataFrame(sc1.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df3

from sklearn.preprocessing import MaxAbsScaler
sc2=MaxAbsScaler()
df4=pd.DataFrame(sc2.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df4

from sklearn.preprocessing import RobustScaler
sc3=RobustScaler()
df5=pd.DataFrame(sc3.fit_transform(df1),columns=['Passenger','Survived','Pclass','Sex','Age','SibSp','Parch','Fare','Embarked'])
df5
```



## OUPUT
### Data.csv :
#### Initial Dataset:
![image](https://user-images.githubusercontent.com/113017853/195593951-c12dbcc0-8eee-4f4d-b7d4-17a1a74cd8e6.png)
## Binary Encoding:
![image](https://user-images.githubusercontent.com/113017853/195594234-b699d675-a5d6-405f-b28f-cfe53f7a73e2.png)

![image](https://user-images.githubusercontent.com/113017853/195594319-1738be3c-b51b-4429-9d03-d87aaa4cd413.png)
### Encoded Dataset:
![image](https://user-images.githubusercontent.com/113017853/195594424-124f9ca4-45a9-46c1-ae72-ec283ae87d8b.png)
### Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/113017853/195594547-e2ba610e-0b4d-4cf5-a3c8-03d14b447857.png)

### Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/113017853/195594654-518d1b8d-9643-4260-9acc-8a77e6772576.png)
### Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/113017853/195594850-83285e75-0d07-4fe0-8575-52ff37ad88d4.png)

### Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/113017853/195594978-02f0d5ac-7eaf-4e03-a80b-6d59895397bd.png)

## Encoding.csv :

### Initial Dataset:
![image](https://user-images.githubusercontent.com/113017853/195595216-158a30ff-a2b5-4327-b4dc-d3083977e299.png)

### Binary Encoding:
![image](https://user-images.githubusercontent.com/113017853/195595339-36284dcf-7055-4f90-87a9-29ee22cb9fc0.png)

![image](https://user-images.githubusercontent.com/113017853/195595371-d721c77f-b9d2-4d6b-b2fc-e4c0daf1c3c0.png)

### Encoded Dataset:
![image](https://user-images.githubusercontent.com/113017853/195595456-a42218d6-ef59-4103-8ac5-89b93db050bc.png)
![image](https://user-images.githubusercontent.com/113017853/195595551-64343355-8929-4c02-ac05-42e2aae1a0bd.png)

### Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/113017853/195595643-0e513edc-9e7c-4d98-b033-406eac0a5bb0.png)

### Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/113017853/195595766-983ca442-b66a-434e-8014-788ca52193ca.png)

### Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/113017853/195595976-4bbab22f-963e-4102-910f-d5b3d3e45956.png)

### Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/113017853/195596114-f3de9eaf-8d7f-4610-a773-638fac5a3e73.png)
## Titanic.csv :
### Initial Dataset:
![image](https://user-images.githubusercontent.com/113017853/195596339-7eac8594-96e8-47ff-8205-2203dc0e059c.png)

### Data cleaning before encoding:
![image](https://user-images.githubusercontent.com/113017853/195596447-5520c31e-72e2-4ace-ac5a-e4cc8c716006.png)

![image](https://user-images.githubusercontent.com/113017853/195596487-bf87dfc5-4f27-4956-ab65-bc57f2942c90.png)

![image](https://user-images.githubusercontent.com/113017853/195596555-4da2169a-5032-47fa-b7f5-deff5c9bfd1a.png)

### Cleaned Dataset:
![image](https://user-images.githubusercontent.com/113017853/195597044-77bf8faf-295e-43f0-9bdb-5f7014cbcae5.png)

### Binary Encoding:
![image](https://user-images.githubusercontent.com/113017853/195597159-807d178a-6d41-4399-99ce-b143551d5105.png)

### Encoded Dataset:
![image](https://user-images.githubusercontent.com/113017853/195597303-1b3f2487-0253-4e0c-8eba-d102d674b508.png)

### Data Scaling using MinMaxScaler:
![image](https://user-images.githubusercontent.com/113017853/195597490-b338e33c-512f-4efa-95db-a8347a971504.png)

### Data Scaling using StandardScaler:
![image](https://user-images.githubusercontent.com/113017853/195597577-6567c11f-60ea-4caf-ba02-928875d65b09.png)

### Data Scaling using MaxAbsScaler:
![image](https://user-images.githubusercontent.com/113017853/195597668-e93b53dd-0014-45e4-84ce-dc7ab84b60c4.png)

### Data Scaling using RobustScaler:
![image](https://user-images.githubusercontent.com/113017853/195597823-69e37a15-67d5-4a93-a056-79acf9c0ab14.png)

## RESULT:

Feature Generation process and Feature Scaling process is applied to the given data frames sucessfully.
