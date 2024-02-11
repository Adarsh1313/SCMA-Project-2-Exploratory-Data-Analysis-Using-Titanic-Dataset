#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


import os


# In[4]:


os.chdir("D:\\Downloads\\Stats Modelling Projects\\Assignment 2")


# In[5]:


os.getcwd()


# In[6]:


titanic = pd.read_csv("D:\\Downloads\\Stats Modelling Projects\\Assignment 2\\train.csv")


# In[7]:


titanic.info()


# In[8]:


titanic.describe()


# In[9]:


titanic.isna().sum()


# ## Conclusion about missing values
# ##### - age has 177 missing values
# ##### - embarked has 2 missing values
# ##### - cabin has a lot of missing values (so we should drop the entire variable)

# In[10]:


titanic.Ticket.describe()


# In[11]:


titanic['Ticket'].value_counts()


# In[12]:


titanic[titanic['Ticket'] == 'CA. 2343' ]


# In[13]:


69.5/11


# In[14]:


titanic.Name.describe()


# In[15]:


titanic.Cabin.describe()


# In[16]:


titanic.Cabin.isnull().sum()


# #### Have to drop Cabin column due to the high null values

# ## 1. Age (Numerical Variable)

# In[17]:


titanic.Age.describe()


# In[18]:


titanic.Age.isnull().sum()


# In[19]:


titanic.Age.plot(kind='hist',bins=80)


# In[20]:


titanic.Age.plot(kind='kde', title="Age KDE Plot")


# In[21]:


titanic.Age.skew()


# In[22]:


titanic.Age.plot(kind='box', title='Age Box Plot')


# ## Conclusion to Age Variable
# ### - age variable has 177 null values and has many outliers so analysis on that cannot be done
# 

# ## 2. Fare (Numerical Variable)

# In[23]:


titanic.Fare.describe()


# In[24]:


titanic.Fare.isnull().sum()


# In[25]:


titanic.Fare.plot(kind='hist',bins=10)


# In[26]:


titanic.Fare.plot(kind='kde', title="Fare KDE Plot")


# In[27]:


titanic.Fare.skew()


# In[28]:


titanic.Fare.plot(kind='box', title='Fare Box Plot')


# ## Conclusion to Fare Variable
# ### - Fare variable has 0 null values but the variable has many outliers as seen in the box plot
# ### - The Fare variable also has a skewness of 4.787316519674893 which suggests most of the data points are concentrated on the left side
# 

# ## 3. SibSp (Numerical Variable)

# In[30]:


titanic.SibSp.describe()


# In[31]:


titanic.SibSp.isnull().sum()


# In[32]:


titanic.SibSp.plot(kind='hist',bins=10)


# In[33]:


titanic.SibSp.plot(kind='kde', title="SibSp KDE Plot")


# In[34]:


titanic.SibSp.skew()


# In[35]:


titanic.SibSp.plot(kind='box', title='SibSp Box Plot')


# ## Conclusion to SibSp Variable
# ### - SibSp variable has 0 null values but the variable has a few outliers as seen in the box plot
# ### - The SibSp variable also has a skewness of 3.6953517271630565 which suggests there are a few instances where the number of siblings or spouses aboard a ship is relatively high.

# ## 4. Parch (Numerical Variable)

# In[36]:


titanic.Parch.describe()


# In[37]:


titanic.Parch.isnull().sum()


# In[38]:


titanic.Parch.plot(kind='hist',bins=20)


# In[39]:


titanic.Parch.plot(kind='kde', title="Parch KDE Plot")


# In[40]:


titanic.Parch.skew()


# In[41]:


titanic.Parch.plot(kind='box', title='Parch Box Plot')


# ## Conclusion to Parch Variable
# ### - Parch variable has 0 null values but the variable has a few outliers as seen in the box plot
# ### - The Parch variable also has a skewness of 2.7491170471010933 which suggests there are some cases where number of parents or children aboard a ship count is relatively high. 

# ## 5. Survived (Categorical Variable)

# In[42]:


titanic.Survived.describe()


# In[43]:


titanic['Survived'].value_counts()


# In[44]:


titanic['Survived'].value_counts().plot(kind='bar')


# In[45]:


titanic['Survived'].value_counts().plot(kind='pie', autopct='%.2f')


# ## Conclusion to Survived Variable
# ### - Survived Variable, being a categorical variable can be visualized through bar and pie charts
# ### - The significant inference from the pie chart is that, 61.62% did not survive whereas the rest 38.38% survived.

# ## 6. Pclass (Categorical Variable)

# In[46]:


titanic.Pclass.describe()


# In[47]:


titanic['Pclass'].value_counts()


# In[48]:


titanic['Pclass'].value_counts().plot(kind='bar')


# In[49]:


titanic['Pclass'].value_counts().plot(kind='pie', autopct='%.2f')


# ## Conclusion to Pclass Variable
# ### - Pclass Variable, being a categorical variable can be visualized through bar and pie charts
# ### - The significant inference from the pie chart is that, more than half (55%) of the passengers were from pclass 3 whereas the rest 45% where evenly distributed from pclass 2 and pclass 1

# ## 7. Sex (Categorical Variable)

# In[50]:


titanic.Sex.describe()


# In[51]:


titanic['Sex'].value_counts()


# In[52]:


titanic['Sex'].value_counts().plot(kind='bar')


# In[52]:


titanic['Sex'].value_counts().plot(kind='pie', autopct='%.2f')


# ## Conclusion to Sex Variable
# ### - Sex Variable, being a categorical variable can be visualized through bar and pie charts
# ### - The significant inference from the pie chart is that, nearly 65% of the passengers were male, whereas the rest 35% were females.

# ## 8. Embarked (Categorical Variable)

# In[53]:


titanic.Embarked.describe()


# In[54]:


titanic.Embarked.isnull().sum()


# In[55]:


titanic['Embarked'].value_counts()


# In[56]:


titanic['Embarked'].value_counts().plot(kind='bar')


# In[57]:


titanic['Embarked'].value_counts().plot(kind='pie', autopct='%.2f')


# ## Conclusion to Embarked Variable
# ### - Embarked Variable, being a categorical variable can be visualized through bar and pie charts - also Embarked has 2 null values, which shouldn't affect the analysis much.
# ### - The significant inference from the pie chart is that, more than half (72%%) of the passengers were from Port S - Southampton whereas the rest 19% and 9% where from Port C - Cherbourg and Port Q - Queenstown respectively.

# In[57]:


titanic.info()


# In[58]:


titanic['Ticket'].value_counts()


# In[59]:


titanic[titanic['Ticket'] == '347082' ]


# In[61]:


titanic[titanic['Ticket'] == 'CA. 2343' ]


# ## Bi Variate Analysis
# #### Survived and Pclass

# In[62]:


pd.crosstab(titanic.Survived, titanic.Pclass, normalize = 'columns')*100


# In[63]:


pd.crosstab(titanic.Survived, titanic.Sex, normalize = 'columns')*100


# ### Conclusion
# 
# **People travelling in Pclass 1 are more likely to survive**
# 
# **Higher class people are more likely to survive than lower class**
# 
# **Females are more likely to survive as compared to men**

# In[64]:


sns.heatmap(pd.crosstab(titanic['Survived'],titanic['Pclass'], normalize='columns')*100)


# In[65]:


sns.heatmap(pd.crosstab(titanic['Survived'],titanic['Sex'], normalize='columns')*100)


# In[66]:


pd.crosstab(index=[titanic['Survived'], titanic['Sex']], columns=titanic['Pclass'], normalize='columns')*100


# In[67]:


sns.heatmap(pd.crosstab(index=[titanic['Survived'], titanic['Sex']], columns=titanic['Pclass'], normalize='columns')*100)


# In[68]:


pd.crosstab(index= titanic['Survived'], columns=[titanic['Sex'],titanic['Pclass']],normalize='columns')*100


# ###### Conclusion
# **- Survival of females more than men in all the classes**

# In[69]:


titanic[titanic['Survived'] == 1]['Age'].plot(kind='kde', label='Dist of Age of people Survived')


# In[70]:


titanic[titanic['Survived'] == 0]['Age'].plot(kind='kde', label='Dist of Age of people Survived')


# In[71]:


titanic[titanic['Survived'] == 1]['Age'].plot(kind='kde', label='Survived')

titanic[titanic['Survived'] == 0]['Age'].plot(kind='kde', label='Not Survived')

plt.legend()
plt.show()


# ###### Conclusion
# **- The "Survived" group has a wider spread of ages than the "Not Survived" group. This suggests that there were more factors that contributed to survival than just age.**

# In[72]:


titanic.columns


# In[73]:


titanic[titanic['Pclass']==1]['Age'].mean()


# In[74]:


titanic[titanic['Pclass']==2]['Age'].mean()


# In[75]:


titanic[titanic['Pclass']==3]['Age'].mean()


# # Feature Engineering

# In[76]:


titanic['Fare'].value_counts()


# In[77]:


# new feature called family_size
titanic['family_size']=titanic['SibSp']+titanic['Parch']+1


# In[78]:


titanic.head(10)


# In[79]:


#indiviual fare columns
titanic['ind_fare'] = titanic['Fare']/titanic['family_size']


# In[80]:


titanic.head(10)


# In[81]:


titanic['Ticket'].value_counts()


# In[82]:


titanic[titanic['Ticket'] == 'CA 2144' ]


# In[83]:


# custom or user defined function
def transform_family_size(num):
    if num == 1:
        return 'alone'
    elif num >1 and num <=5:
        return 'small family'
    else:
        return 'large family'


# In[84]:


titanic['family_type']= titanic['family_size'].apply(transform_family_size)


# In[85]:


titanic['family_type']


# In[87]:


titanic[titanic['family_size']==3]


# In[88]:


pd.crosstab(titanic.Survived, titanic.family_type, normalize = 'columns')*100


# ### Conclusion
# **- The likelihood of a passenger surviving in a small family is higher (56%) as compared to passengers travelling alone and in a large family which is 30% and 14% respectively.**

# In[124]:


# custom or user defined function
def age_trans(num):
    if  0< num <=17:
        return 'child'
    elif 17< num <= 60:
        return 'adults'
    else:
        return 'seniors'


# In[125]:


titanic['Age_group'] = titanic['Age'].apply(age_trans)


# In[126]:


pd.crosstab(titanic.Survived, titanic.Age_group, normalize = 'columns')*100


# ### Conclusion
# **- The likelihood of a passenger surviving who is a child is higher (53%) as compared to passengers travelling who are adults and seniors which are 38% and 28% respectively.**

# In[127]:


titanic.info()


# In[128]:


titanic['family_type'].unique()


# In[129]:


titanic['family_type'].value_counts()


# In[130]:


titanic['Age_group'].unique()


# In[131]:


titanic['Age_group'].value_counts()


# In[132]:


titanic['Survived'].unique()


# # Modelling the Dataset

# In[133]:


import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[134]:


titanic.info()


# In[135]:


df = titanic.drop(columns=['Name', 'PassengerId','Cabin','Ticket','Embarked','SibSp'], axis=1)


# In[136]:


df.head(5)


# In[137]:


df.shape


# In[138]:


df = df.dropna()


# In[139]:


df.shape


# In[140]:


df.info()


# In[141]:


df.isnull().sum()


# In[142]:


pd.get_dummies(df['Sex']).head(5)


# #### Converting Sex Column to numerical data from categorical

# In[143]:


#function to transform the Sex column
def gender_transform(val):
    if val == 'male':
        return 1
    else:
        return 0


# In[144]:


binary_sex = df['Sex'].apply(gender_transform)


# In[145]:


binary_sex


# #### Converting Pclass Column to numerical data from categorical

# In[108]:


pd.get_dummies(df['Pclass']).head(5)


# In[172]:


binary_Pclass = pd.get_dummies(df['Pclass'],drop_first = True)


# In[173]:


binary_Pclass


# In[177]:


binary_Pclass.info()


# In[188]:


def class_transform(val):
    return val.eq(True).astype(int)


# In[181]:


binary_Pclass = binary_Pclass.apply(class_transform)


# In[191]:


binary_Pclass


# #### Converting Family Type Column to numerical data from categorical

# In[192]:


pd.get_dummies(df['family_type']).head(5)


# In[193]:


binary_family_type = pd.get_dummies(df['family_type'],drop_first = True)


# In[194]:


binary_family_type


# In[195]:


binary_family_type.info()


# In[196]:


def class_transform(val):
    return val.eq(True).astype(int)


# In[197]:


binary_family_type = binary_family_type.apply(class_transform)


# In[198]:


binary_family_type


# #### Converting Age Group Column to numerical data from categorical

# In[199]:


pd.get_dummies(df['Age_group']).head(5)


# In[200]:


binary_Age_group = pd.get_dummies(df['Age_group'],drop_first = True)


# In[201]:


binary_Age_group


# In[202]:


binary_Age_group.info()


# In[204]:


def class_transform(val):
    return val.eq(True).astype(int)


# In[205]:


binary_Age_group = binary_Age_group.apply(class_transform)


# In[206]:


binary_Age_group


# In[208]:


df.info()


# In[210]:


df.head()


# In[211]:


modified_data_set = pd.concat([df, binary_sex, binary_Pclass,binary_family_type,binary_Age_group], axis = 1)


# In[212]:


modified_data_set.head()


# In[257]:


final_data_set = modified_data_set.drop(columns = ['Sex', 'Pclass','family_size','family_type','Age_group',])


# In[258]:


final_data_set.rename(columns = {2:'pcalss2',3:'pclass3'}, inplace = True)


# In[259]:


final_data_set


# In[260]:


final_data_set.isna().sum()


# ### Model Building 

# In[261]:


Y= final_data_set['Survived']
X= final_data_set.drop(['Survived'], axis = 1 )


# In[262]:


Y.info()


# In[263]:


X.info()


# In[264]:


X.shape


# In[265]:


#train sample
714 * 0.8


# In[266]:


#test sample
714 *.2


# In[267]:


test_set_size = 0.2
seed = 1


# In[268]:


#test train split
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X,Y, test_size = test_set_size , random_state = seed)


# In[269]:


#load libraries
import pandas as pd
import sklearn
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import seaborn as sns
import matplotlib.pyplot
get_ipython().run_line_magic('matplotlib', 'inline')
import math


# In[270]:


#initiating the model
model = LogisticRegression()


# In[271]:


type(model)


# In[272]:


type(titanic)


# In[273]:


#fitting the model
model.fit(X_train,Y_train)


# In[274]:


predictions = model.predict(X_test)


# In[275]:


len(predictions)


# In[276]:


report = classification_report(Y_test, predictions)


# In[277]:


print(report)


# In[278]:


from sklearn import metrics
confusion_matrix = metrics.confusion_matrix(Y_test, predictions)


# In[279]:


confusion_matrix


# In[280]:


accuracy = (75+36)/(75+11+21+36)
print(accuracy)


# In[281]:


cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.show()


# In[282]:


precision = (75)/(75+21)
print(precision)


# In[283]:


recall = (75)/(75+36)
print(recall)

