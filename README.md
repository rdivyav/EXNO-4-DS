# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/bmi.csv")
df.head()
```

![Screenshot 2024-10-17 095758](https://github.com/user-attachments/assets/c0a49cef-cffe-4d94-9bab-0b6a98f34018)

```
df.dropna()
```

![Screenshot 2024-10-17 095808](https://github.com/user-attachments/assets/0a7909b0-b884-4ec0-b54e-1295d84dbb94)

```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
199
```
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```

![Screenshot 2024-10-17 095831](https://github.com/user-attachments/assets/07e37ad5-bb93-41e1-bc76-c0b8d6188d78)

```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```

![Screenshot 2024-10-17 095856](https://github.com/user-attachments/assets/1d9f4e0b-e80f-48aa-9994-48d91c2d8bcc)

```
df3=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df3[['Height','Weight']]=scaler.fit_transform(df3[['Height','Weight']])
df3
```

![Screenshot 2024-10-17 095911](https://github.com/user-attachments/assets/dcade62c-7257-4f44-bc24-a4f278928dc3)

```
df4=pd.read_csv("/content/bmi.csv")
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df4[['Height','Weight']]=scaler.fit_transform(df4[['Height','Weight']])
df4.head()
```

![Screenshot 2024-10-17 095924](https://github.com/user-attachments/assets/3878ba6f-f54e-4173-863d-e8916e0b1b97)

```
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import chi2
df=pd.read_csv('/content/titanic_dataset.csv')
df.columns
```

![Screenshot 2024-10-17 095940](https://github.com/user-attachments/assets/3e2212f9-6f45-4e6b-9ff4-bd9109c7f96a)

```
df.shape
```

![Screenshot 2024-10-17 095951](https://github.com/user-attachments/assets/6003efd3-b9e9-4728-878b-9e52bbeb4ad5)

```
df1=df.drop(["Name","Sex","Ticket","Cabin","Embarked"],axis=1)
df1.columns
```

![Screenshot 2024-10-17 095958](https://github.com/user-attachments/assets/1091ebab-785a-4cee-bda6-b00d38c0c0f2)

```
df1['Age'].isnull().sum()
```

![Screenshot 2024-10-17 100012](https://github.com/user-attachments/assets/ab3407ea-9c4e-48ed-b1f9-5e4744c05db3)

```
df1['Age'].fillna(method= 'ffill')
```

![Screenshot 2024-10-17 100025](https://github.com/user-attachments/assets/27f0c7f3-1046-4638-9cbb-d13e1630f2d6)

```
df1['Age']=df1['Age'].fillna(method='ffill')
df1['Age'].isnull().sum()
```

![Screenshot 2024-10-17 100049](https://github.com/user-attachments/assets/135fd06e-bca2-4fb0-8e0b-eb5e73212d07)

```
feature=SelectKBest(mutual_info_classif,k=3)
df1.columns
```

![Screenshot 2024-10-17 100116](https://github.com/user-attachments/assets/06009f2a-71b0-4f25-a6a9-5f3bb4f7fae7)

```
df1.columns
```

![Screenshot 2024-10-17 100130](https://github.com/user-attachments/assets/094e0936-d4b9-42bd-86e8-a0ae720d09c1)

```
X=df1.iloc[:,0:6]
y=df1.iloc[:,6]
X.columns
```

![Screenshot 2024-10-17 100346](https://github.com/user-attachments/assets/d19bfd7c-00fc-4249-928d-85694d49e78b)

```
import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
data=pd.read_csv('/content/titanic_dataset.csv')
data=data.dropna()
X=data.drop(['Survived','Name','Ticket'],axis=1)
y=data['Survived']
X
```

![Screenshot 2024-10-17 100415](https://github.com/user-attachments/assets/20a5bc79-1f74-4b6d-814d-478cabceb5df)
```

data["Sex"]=data["Sex"].astype("category")
data["Cabin"]=data["Cabin"].astype("category")
data["Embarked"]=data["Embarked"].astype("category")

data["Sex"]=data["Sex"].cat.codes
data["Cabin"]=data["Cabin"].cat.codes
data["Embarked"]=data["Embarked"].cat.codes
data
```

![Screenshot 2024-10-17 100431](https://github.com/user-attachments/assets/215870d6-2849-4537-ba82-6982038f8bd8)
```

X.info()
```

![Screenshot 2024-10-17 100445](https://github.com/user-attachments/assets/0c0d1bf6-1d13-4962-a7b1-6f617a70fad4)
```

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
df= pd.read_csv('/content/titanic_dataset.csv')
df.columns
```

![Screenshot 2024-10-17 111325](https://github.com/user-attachments/assets/c52a19ad-def3-4146-ad39-cd432ea7a84b)
```
df
```

![Screenshot 2024-10-17 111336](https://github.com/user-attachments/assets/56894349-20a9-4ecd-bc01-b22eea4cefe8)

```
df = df.dropna()
df.isnull().sum()
```

![Screenshot 2024-10-17 111347](https://github.com/user-attachments/assets/0801609b-c8b3-4e9d-b12b-c0fe8612f4f2)

```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips = sns.load_dataset('tips')
tips.head()
```

![Screenshot 2024-10-17 111355](https://github.com/user-attachments/assets/bebd9760-5bf2-4116-8db5-aa67e8cbe1a8)

       
# RESULT:
       Thus the feature scaling and selection has been performed for the given data sets.
