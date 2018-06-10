
# coding: utf-8

# In[1]:


import tensorflow as tf
print ("TensorFlow version: " + tf.__version__)
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from keras.models import Sequential  
from keras.layers import Dense, Activation, Conv2D  
from keras.layers import MaxPool2D, Flatten, Dropout, ZeroPadding2D, BatchNormalization  
from keras.utils import np_utils  
import keras
from keras.models import save_model, load_model  
from keras.models import Model


# In[2]:


train = pd.read_csv('training_set.csv', header=None, sep=',')
print('dataset shape {}'.format(train.shape))
train.head()


# In[3]:


X_train = train.iloc[:, 0:18]
y_train = train.iloc[:, 18]
print('shape of X {}; shape of y {}'.format(X_train.shape, y_train.shape))

# 绘制计数直方图
sns.countplot(y_train)
plt.show()
# 使用pd.Series.value_counts()
print(y_train.value_counts())


# In[4]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)


# In[29]:


y_train[:10]


# In[6]:


y_trainOneHot = np_utils.to_categorical(y_train)


# In[7]:


y_trainOneHot[:15]


# In[8]:


model = Sequential()


# In[9]:


model.add(Dense(units=18, input_dim=18, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=18, kernel_initializer='normal', activation='relu'))
model.add(Dense(units=12, kernel_initializer='normal', activation='relu'))


# In[10]:


model.add(Dense(units=6, kernel_initializer='normal', activation='softmax'))


# In[11]:


print(model.summary())


# In[12]:


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[13]:


train_history = model.fit(x=X_train_scaled, y=y_trainOneHot, validation_split=0.25, epochs=50, batch_size=20, verbose=2)


# In[14]:


def show_train_history(train_history, train, validation):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


# In[15]:


show_train_history(train_history, 'acc', 'val_acc')


# In[16]:


show_train_history(train_history, 'loss', 'val_loss')


# In[17]:


scores = model.evaluate(X_train_scaled, y_trainOneHot)
print()
print('accuracy=',scores[1])


# In[18]:


test = pd.read_csv('testing_set.csv', header=None, sep=',')
print('dataset shape {}'.format(test.shape))
test.head()


# In[19]:


X_test = test.iloc[:, 0:18]
y_test = test.iloc[:, 18]
print('shape of X {}; shape of y {}'.format(X_test.shape, y_test.shape))

# 绘制计数直方图
sns.countplot(y_test)
plt.show()
# 使用pd.Series.value_counts()
print(y_test.value_counts())


# In[20]:


scaler.fit(X_test)
X_test_scaled = scaler.transform(X_test)


# In[24]:


y_testOneHot = np_utils.to_categorical(y_test)


# In[26]:


test_scores = model.evaluate(X_test_scaled, y_testOneHot)
print()
print('accuracy=', test_scores[1])

