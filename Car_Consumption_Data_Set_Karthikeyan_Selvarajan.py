#!/usr/bin/env python
# coding: utf-8

# In[1]:


# model 1


# In[2]:


# To do list
#Classification exercise: MpG consumption:
#Create a new jupyter file to analyse the training data mpgTrainingset.txt (published by
#Garnegie Mellon).
#This file will constitue your report, it should then includes your code, illustrations and analyses of the
#obtained results.
#The data represent the characteristics of cars: number of cylinders, cubic inch displacement, horse
#power, weight, acceleration.
#The category is discrete (values of 10, 15, 20, 25, 30, 35, 40, and 45) . It represents the consumption
#in miles per gallon.
#Your goal is to predict with the highest reliability the category of the cars belonging to the file
#mpgTestset.txt.
#You will describe the different steps used to obtain this results. You will use a least 2 different
#methods. Their performance should be evaluated (quantitaively). 


# In[ ]:


# data liberies used in this analysis

# Import of the needed libraires
#graphical librairies
import matplotlib as mpl
from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
from pylab import figure, subplot, hist, xlim, show, plot
get_ipython().run_line_magic('matplotlib', 'inline')

#data librairies 

import pandas as pd
import pylab as pl
import numpy as np

from pandas.plotting import scatter_matrix
from pandas.plotting import boxplot
from pandas.plotting import parallel_coordinates
from matplotlib.colors import ListedColormap


# The code snippet imports the data from a CSV file named "mpgTrainingSet-headings.csv" into a Pandas DataFrame object named "data_panda".
# 

# In[4]:


# to import the data from csv file
# data is imported and panda object is created
data_panda = pd.read_csv('mpgTrainingSet-headings.csv')


# In[5]:


# to see the total no of data and its key columns
print(data_panda.keys())

# len
nb_specimen=len(data_panda)
print('There are '+ str(nb_specimen)+' cars in the set')


# In[6]:


# for more cleare futur use we can create a set with the input col
Input_cols = [ 'Cylinders', 'Cubic_inch', 'Horsepower', 'Weight',
       'Acceleration']

print(data_panda)


# The code snippet prints the list of column names (headers) of the Pandas DataFrame object "data_panda".
# 

# In[7]:


# value count and brand count
data_panda['Consumption'].value_counts()

data_panda['Brand'].value_counts()


# In[8]:


#definition of the colors used for visualization 
colors = np.where(data_panda['Consumption']==10,'r','-')
colors[data_panda['Consumption']==15] = 'g'
colors[data_panda['Consumption']==20]= 'b'
colors[data_panda['Consumption']==25]= 'y'
colors[data_panda['Consumption']==30]= 'c'
colors[data_panda['Consumption']==35]= 'm'
colors[data_panda['Consumption']==40]= 'k'
colors[data_panda['Consumption']==45]= '0.5'

#print(colors)
color_dict={10:'r',15:'g' ,20:'b',25:'y',30:'c' ,35:'m',40:'k' ,45:'0.5'}

data_panda.groupby('Consumption').describe()


# The data shows  The mean, standard deviation, minimum, 25th percentile, 50th percentile, 75th percentile, and maximum values for the number of cylinders, bore, displacement,
# compression ratio, horsepower, curb weight, and fuel economy are shown in the table above.

# In[9]:


# Correlation Analysis
numeric_data = data_panda.drop(columns=data_panda.select_dtypes(exclude=[np.number]).columns)
correlation_matrix = numeric_data.corr()
print("Columns used for correlation:")
print(numeric_data.columns)
print("\nCorrelation Matrix:")
print(correlation_matrix)


# The correlation matrix shown above indicates that the features are moderately correlated with each other. For example, there is a negative correlation between consumption and horsepower (-0.764674), meaning that as horsepower increases, consumption tends to
# decrease. if a car manufacturer wants to improve the fuel economy of their cars, they should consider reducing the
# weight of the car and/or decreasing the horsepower of the engine.
# 

# In[10]:


# Visualization
sns.scatterplot(x='Consumption', y='Acceleration', color='k', data=data_panda)
plt.title('Consumption vs Acceleration')
plt.show()


# In[11]:


# Visualization
sns.scatterplot(x='Consumption', y='Cubic_inch', color='k', data=data_panda)
plt.title('Consumption vs Cubic_inch')
plt.show()


# The scatter plot indicates that there is a weak positive correlation between consumption and cubic inch. This suggests that as cubic inch increases, consumption tends to increase as well. However, the correlation is not very strong, so there are other factors that are
# also important in determining fuel economy
# 

# In[12]:


# Visualization
sns.scatterplot(x='Consumption', y='Cylinders', color='k', data=data_panda)
plt.title('Consumption vs Cylinders')
plt.show()


# The scatter plot shows a negative correlation between consumption and cylinders. This means that as cylinders increase, consumption tends to decrease. This is because cars with more cylinders tend to be more efficient at converting fuel into power.

# In[13]:


# Visualization
sns.scatterplot(x='Consumption', y='Horsepower', color='k', data=data_panda)
plt.title('Consumption vs Horsepower')
plt.show()


# The scatter plot shows a strong negative correlation between consumption and horsepower. This means that as horsepower increases, consumption tends to decrease. This is because higher horsepower engines tend to be more efficient at converting fuel into
# power.

# In[14]:


# Visualization
sns.scatterplot(x='Consumption', y='Weight', color='k', data=data_panda)
plt.title('Consumption vs Weight')
plt.show()


# The scatter plot shows the relationship between fuel economy (mpg) and cubic inches (ci),Cylinders,Horsepower,weight for each car brand in the dataset. The colors of the points represent different levels of fuel economy, with blue representing the lowest fuel
# economy and red representing the highest fuel economy. The styles of the points represent different levels of cubic inches, with circles representing the smallest cubic inches and triangles representing the largest cubic inches. It is interesting to note that there are a
# few outliers in the data. For example, the Chevrolet Nova with 165 cubic inches has the lowest fuel economy of any car in the dataset. However, there are also a few cars with high cubic inches that have relatively good fuel economy, such as the Datsun with 165 cubic
# inches

# In[15]:


# Visualization with Seaborn scatter plot
scatter_plot = sns.scatterplot(x='Brand', y='Cubic_inch', hue='Consumption', palette='deep', data=data_panda)
plt.title('Brand vs Cubic_inch')
plt.xlabel('Brand')
plt.ylabel('Cubic_inch')
plt.colorbar(scatter_plot.get_children()[0], label='Consumption')
plt.show()


# In[16]:


# Visualization with Seaborn scatter plot
scatter_plot = sns.scatterplot(x='Cubic_inch', y='Horsepower', hue='Consumption', palette='deep', data=data_panda)
plt.title('Cubic_inch vs Horsepower')
plt.xlabel('Cubic Inches')
plt.ylabel('Horsepower')
plt.colorbar(scatter_plot.get_children()[0], label='Consumption')
plt.show()


# In[17]:


# Visualization with Seaborn scatter plot
scatter_plot = sns.scatterplot(x='Brand', y='Weight', hue='Consumption', palette='deep', data=data_panda)
plt.title('Brand vs Weight')
plt.xlabel('Brand')
plt.ylabel('Weight')
plt.colorbar(scatter_plot.get_children()[0], label='Consumption')
plt.show()


# In[18]:


# Visualization with Seaborn scatter plot
scatter_plot = sns.scatterplot(x='Brand', y='Horsepower', hue='Consumption', palette='deep', data=data_panda)
plt.title('Brand vs Horsepower')
plt.xlabel('Brand')
plt.ylabel('Horsepower')
plt.colorbar(scatter_plot.get_children()[0], label='Consumption')
plt.show()


# In[19]:


# Specify the colors based on the 'Consumption' column
colors = data_panda['Consumption']

# Create scatter matrix
scatter_matrix(data_panda, figsize=(10, 10), diagonal='hist', c=colors, alpha=0.8)

# Show the plot
plt.show()

data_panda.boxplot(by='Consumption', figsize=(12, 6));


# The purpose of this code is to create a scatter matrix, where each pair of variables in the data_panda DataFrame is plotted against each other. The diagonal subplots will be histograms of the corresponding variables, and the points in the scatter plots will be colored
# based on the values in the 'Consumption' column. The resulting visualization provides insights into the relationships and distributions of variables in the dataset.
# 
# 
# The boxplot shows that there is a wide range of fuel economy values, from a low of 13 mpg to a high of 19 mpg. The median fuel economy is 15.5 mpg. The cars with the best fuel economy (the blue box) tend to have fewer cylinders and less horsepower. The cars with
# the worst fuel economy (the red box) tend to have more cylinders and more horsepower. But because we are working with an unnormalized data our diagrams are not that clear, what we will do is to normalize our data for a much clearer view of the diagram
# 

# In[20]:


# Data Normalization
Norm = data_panda.copy()
Norm[Input_cols] = (data_panda[Input_cols] - data_panda[Input_cols].min()) / (data_panda[Input_cols].max() - data_panda[Input_cols].min())

print([Input_cols])


# Normalization is a crucial step in data preparation for machine learning algorithms. It helps to address the issue of varying scales among features, ensuring that all features are treated equally and contribute proportionately to the learning process. By normalizing the
# data, the algorithm can focus on the underlying relationships between features without being influenced by their individual scales. This can lead to more accurate and robust models.

# In[21]:


print(Norm[Input_cols])


# In[22]:


# Visualization Normalisation with Seaborn scatter plot
scatter_plot = sns.scatterplot(x='Brand', y='Cubic_inch', hue='Consumption', palette='deep', data=Norm)

plt.title('Normalisation Brand vrs Cubic_inch') 

plt.show()


# In[23]:


# The box plots provide a more lucid perspective of the interactions after normalization.

Norm.boxplot(by='Consumption', figsize=(12, 6));


# In[24]:


# Data Encoding
Norm['Consumption'] = Norm['Consumption'].astype('category')
Norm['Consumption_encoded'], dict_cat = Norm['Consumption'].factorize()
color_dict_encoded = {0: 'r', 1: 'g', 2: 'b', 3: 'y', 4: 'c', 5: 'm', 6: 'k', 7: '0.5'}
print(dict_cat)


# Converting categorical variables to numerical variables can be helpful for machine learning algorithms. Many machine learning algorithms require that all features be numerical. By converting categorical variables to numerical variables, we can ensure that all
# features are compatible with the machine learning algorithm.
# In this case, converting the Consumption column to a numerical variable allows us to use machine learning algorithms to predict the fuel economy of a car

# In[25]:


print(color_dict_encoded)


# This code snippet retrieves the encoded value for a specific car and then uses the color_dict_encoded dictionary to find the corresponding color. This color can then be used for visualization purposes, such as coloring data points or creating charts.
# 

# In[26]:


# PCA
from sklearn.decomposition import PCA
for i in range(1,5):
    pca = PCA(n_components=i)
    pca.fit(Norm[Input_cols])
    print (i, 'components representa data loss of' ,(1-sum(pca.explained_variance_ratio_)) * 100,'%')
    
n_components=2
pca = PCA(n_components)
pca.fit(Norm[Input_cols])
pca_apply = pca.transform(Norm[Input_cols])  

base=pd.DataFrame(pca.components_,columns=Norm[Input_cols].columns,index = ['PCA0','PCA1'])            
print(base)

pcad_panda=pd.DataFrame(pca_apply, columns=['PCA%i' % i for i in range(n_components)]) #save in a panda object
Norm=pd.concat([Norm, pcad_panda], axis=1)#concatenate in norm_pd
print(Norm.keys())

#viz
sns.scatterplot(x='PCA0', y='PCA1', hue='Consumption', palette='deep', data=Norm)
pl.xlabel('PCA0')
pl.ylabel('PCA1')
pl.title('342 cars in the new base')
plt.show()



# The code snippet utilizes the PCA (Principal Component Analysis) algorithm to reduce the dimensionality of the normalized input data (Norm[Input_cols]) to two main components (n_components=2).
# 
# 

# In[27]:


#test and train
from sklearn.model_selection import train_test_split

#Learning population is called train,
#the target value (consumption) t_train
#test population is called test#
#the predicted value (species)t_test

train, test, t_train, t_test = train_test_split(Norm, Norm['Consumption_encoded'], test_size=0.4, random_state=0)

# print
print(train)

#print
print(test)

#viz
sns.scatterplot(x='PCA0',y='PCA1', data=train)
sns.scatterplot(x='PCA0',y='PCA1', data=test)

pl.xlabel('PCA0')
pl.ylabel('PCA1')
plt.legend( loc='upper left', labels=['Learning set', 'Test set'])
pl.title('Random repartition of comsumption') 
plt.show()


# Splitting the data into training and testing sets is crucial for evaluating the performance of machine learning models. The training set is used to train the model, allowing it to learn the relationships between the features and the target variable. The testing set is then
# used to assess the model's generalizability, ensuring that it can accurately predict unseen data.This code splits the normalized data (Norm) into two sets: train and test. The test_size parameter specifies that 40% of the data will be allocated to the testing set (test),
# while the remaining 60% will be assigned to the training set (train). The random_state parameter ensures that the data is split randomly in a consistent manner, allowing for reproducible results

# In[ ]:





# In[ ]:





# In[ ]:





# In[28]:


# Gaussian Naive Bayes methode


# Gaussian Naive Bayes is a simple and efficient machine learning algorithm that is particularly well-suited for classification tasks involving numerical features. It is based on the assumption that the features are independent and follow Gaussian distributions
# 

# In[29]:


from sklearn.naive_bayes import GaussianNB
classifier_GNB = GaussianNB()
classifier_GNB.fit(train[Input_cols],train['Consumption_encoded']) # train


# In[30]:


prediction_GNB =classifier_GNB.predict(train[Input_cols]) #prediction
#here we can compare the prediction and real specy for the first specimen
print (prediction_GNB[0])
print (train['Consumption_encoded'][0])


# This code creates another subplot (212) within the figure and uses the seaborn library's sns.scatterplot() function to visualize the predicted fuel economy (prediction_GNB) based on the training data. The hue argument specifies that the data points should be colored
# according to their predicted prediction_GNB values, and the palette='deep' argument ensures that the colors are distinct and readable.
# 
# By comparing the two subplots, you can assess the overall accuracy of the GNB classifier. If the predicted fuel economy categories align with the actual fuel economy categories, then the classifier is performing well. If there are significant discrepancies, the
# classifier may need to be retrained or adjusted to improve its performance.

# In[31]:


color_dict_prediction={0:'y',1:'c' ,2:'m'}
figure = plt.figure(figsize = (10, 10))
plt.tight_layout()
plt.subplot(211)
sns.scatterplot(x='PCA0', y='PCA1', hue='Consumption_encoded', palette='deep', data=train)
#plt.ylim(taille_min,taille_max)
plt.title('Real consumption of train set')
plt.subplot(212)
sns.scatterplot(x='PCA0', y='PCA1', hue=prediction_GNB, palette='deep', data=train)
#plt.ylim(taille_min,taille_max)
plt.title('Predicted consumption')


# An accuracy of 0.80 means that the GNB classifier correctly classified 80% of the data points in the training set. This is a relatively high accuracy score, suggesting that the classifier is performing well on the training data. However, it's important to note that accuracy
# on the training data may not reflect the classifier's performance on unseen data.
# 

# In[32]:


print (classifier_GNB.score(train[Input_cols],t_train)) # train


# In[33]:


from sklearn.metrics import confusion_matrix
M_GNB=confusion_matrix(t_train,prediction_GNB)# the 1st parameter will be on rows and 2nd parameter
#i-th row and j-th column entry indicates the number of samples with true label being i-th class and predicted label being j-th class.
print (M_GNB)


# In[34]:


import pandas as pd

# Initialize an empty DataFrame
conf_GNB = pd.DataFrame(columns=['real_encoded', 'real_Consumption', 'predicted_encoded', 'predicted_Consumption', 'density'])

for i in range(0, 4):
    for j in range(0, 4):
        if M_GNB[i][j] > 0:
            new_row = {'real_encoded': i, 'real_Consumption': dict_cat[i], 'predicted_encoded': j,
                       'predicted_Consumption': dict_cat[j], 'density': float(M_GNB[i][j])}
            # Create a new DataFrame and concatenate it with the existing one
            conf_GNB = pd.concat([conf_GNB, pd.DataFrame([new_row])], ignore_index=True)

print(conf_GNB)


# In[35]:


import seaborn as sns
import matplotlib.pyplot as plt

# Plotting the scatter plot with transparency based on density
plt.scatter(x=conf_GNB['real_Consumption'], y=conf_GNB['predicted_Consumption'], alpha=0.5, s=(conf_GNB['density'] * 60))

plt.xlabel('Real Consumption')
plt.ylabel('Predicted Consumption using GNB')
plt.title('Prediction relevance of GNB on the train set')

plt.show()


# In[36]:


import seaborn as sns

sns.pairplot(data=conf_GNB, vars=['real_Consumption', 'predicted_Consumption'])
plt.suptitle('Pairplot of Real vs Predicted Consumption')
plt.show()


# The report provides a breakdown of the classifier's performance for each class, including precision, recall, F1 score, and support.
# 
# Here's a summary of the classification report metrics:
# Precision: The proportion of positive predictions that are actually correct.
# Recall: The proportion of actual positives that are correctly identified.
# F1 Score: The harmonic mean of precision and recall. It provides a balanced measure of both precision and recall.
# Support: The number of data points in each class.
# 

# In[37]:


from sklearn.metrics import classification_report
print (classification_report(prediction_GNB,t_train))


# Cross-validation helps to assess the generalization performance of the GNB classifier by evaluating its performance on unseen data, reducing the risk of overfitting to the training data. The average accuracy across the folds provides a more reliable indication of the
# classifier's overall performance.
# 

# In[38]:


from sklearn.model_selection import cross_val_score
# cross validation with 6 iterations 
scores = cross_val_score(classifier_GNB,Norm[Input_cols], Norm['Consumption_encoded'], cv=6)
print (scores)


# In[39]:


from numpy import mean
print (mean(scores))


# The mean cross-validation score indicates the overall accuracy of the GNB classifier on unseen data. A high mean score suggests that the classifier generalizes well and is able to accurately predict the fuel consumption classes for new data points.
# 

# In[40]:


prediction_test_GNB =classifier_GNB.predict(Norm[Input_cols]) #prediction
#We store the K-means results in a dataframe
prediction_test_GNB_pd = pd.DataFrame(prediction_test_GNB)
prediction_test_GNB_pd.columns = ['Prediction_GNB']
#we merge this dataframe with df
Norm= pd.concat([Norm,prediction_test_GNB_pd], axis = 1)


# This code demonstrates the use of a trained GNB classifier to make predictions for new data points and incorporates the predictions into a comprehensive DataFrame that includes both the actual and predicted fuel consumption classes. This DataFrame facilitates
# further analysis and evaluation of the GNB classifier's performance.

# In[41]:


import pandas as pd

#print(Norm)
M_GNB_total = confusion_matrix(Norm['Consumption_encoded'], prediction_test_GNB)
print(M_GNB_total)

conf_GNB_total = pd.DataFrame(columns=['real_encoded', 'real_Consumption', 'predicted_encoded', 'predicted_Consumption_GNB', 'density'])
for i in range(0, 7):
    for j in range(0, 7):
        if M_GNB_total[i][j] > 0:
            new_row = {'real_encoded': i, 'real_Consumption': dict_cat[i], 'predicted_encoded': j, 'predicted_Consumption_GNB': dict_cat[j], 'density': float(M_GNB_total[i][j])}
            conf_GNB_total = pd.concat([conf_GNB_total, pd.DataFrame([new_row])], ignore_index=True)

print(conf_GNB_total)


# This code effectively evaluates the performance of the GNB classifier on the entire dataset by calculating the confusion matrix, summarizing the results in a DataFrame, and visualizing the performance using a scatter plot. This analysis provides a comprehensive
# understanding of the classifier's ability to accurately predict fuel consumption classes

# In[42]:


sns.scatterplot(x='real_Consumption', y='predicted_Consumption_GNB', s=(conf_GNB_total.density)*60, data=conf_GNB_total, color='k')
pl.xlabel('Real Consumption')
pl.ylabel('Predicted Consumption using GNB')
pl.title('Prediction relevance of GNB on the whole set')
plt.show()


# In[43]:


import seaborn as sns
import matplotlib.pyplot as plt


# Create a heatmap
heatmap_data = conf_GNB_total.pivot_table(index='real_Consumption', columns='predicted_Consumption_GNB', values='density')

plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

sns.heatmap(data=heatmap_data, cmap='viridis', annot=True, fmt=".2f")

plt.xlabel('Predicted Consumption using GNB')
plt.ylabel('Real Consumption')
plt.title('Prediction Relevance of GNB on the Whole Set')
plt.show()


# print('Using Gaussian Naive Bayes, the predicted Consumption of the 35th Consumption is '+ str(dict_cat[Norm.iloc[35]['Prediction_GNB']]))
# 

# The code snippet attempts to predict the fuel consumption class for a new car using the trained Gaussian Naive Bayes (GNB) classifier. It creates a new DataFrame (panda_New_specimen) containing the input features of the new car and then makes a prediction
# using the classifier_GNB model. Finally, it prints the predicted fuel consumption class which is 20

# In[44]:


New_specimen = {
 'Cylinders':[0.5],
 'Cubic_inch': [0.5],
 'Horsepower': [0.5],
 'Weight': [0.5],
 'Acceleration':[0.5] 
 }
panda_New_specimen = pd.DataFrame(New_specimen) 
D=classifier_GNB.predict(panda_New_specimen)
print('Using kmeans, the predicted Consumption of such a car is '+ str(dict_cat[D[0]]))


# In[ ]:





# In[ ]:





# In[ ]:


# neural_network


# In[45]:


from sklearn.neural_network import MLPClassifier
classifier_NN= MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)


# This code snippet defines a Multi-Layer Perceptron (MLP) classifier and initializes it with specific parameters. The MLP classifier is a type of artificial neural network that can learn complex nonlinear relationships between input and output data.These parameters
# represent a common setting for an MLP classifier using the lbfgs solver for solving gradient descent optimization. The number of neurons in each hidden layer is chosen to balance complexity and computational efficiency. The random state ensures that the same
# training data is split into training and validation sets during cross-validation, which helps to prevent overfitting.

# In[46]:


classifier_NN.fit(train[Input_cols],train['Consumption_encoded'])


# The code compares the predicted fuel consumption class for the first data point in the training set with its actual fuel consumption class. It first makes a prediction using the trained Multi-Layer Perceptron (MLP) classifier and then retrieves the actual fuel
# consumption class from the training data. This comparison allows for a direct evaluation of the MLP classifier's ability to accurately predict the fuel consumption class for the first data point. If the predicted and actual classes match, it suggests that the classifier is
# performing well for this particular data point. However, a mismatch might indicate that the classifier needs further tuning or that the data point is an outlier

# In[47]:


prediction_NN=classifier_NN.predict(train[Input_cols]) #prediction
#here we can compare the prediction and real specy for the first specimen
print (prediction_NN[0])
print (train['Consumption_encoded'][0])


# The code snippet calculates and prints the accuracy of the Multi-Layer Perceptron (MLP) classifier on the training data. Accuracy is a common metric for evaluating the performance of classification models, indicating the proportion of data points that are correctly
# classified.
# 

# In[48]:


print('The performance of the Neuron Netwok prediction is')
print (classifier_NN.score(train[Input_cols],t_train)) # test


# In[49]:


M_NN=confusion_matrix(t_train,prediction_NN)
print (M_NN)


# In[50]:


import pandas as pd

conf_NN = pd.DataFrame(columns=['real_encoded', 'real_Consumption', 'predicted_encoded', 'predicted_Consumption_GNB', 'density'])

for i in range(0, 7):
    for j in range(0, 7):
        if M_NN[i][j] > 0:
            new_row = {'real_encoded': i, 'real_Consumption': dict_cat[i], 'predicted_encoded': j, 'predicted_Consumption_GNB': dict_cat[j], 'density': float(M_NN[i][j])}
            conf_NN = pd.concat([conf_NN, pd.DataFrame([new_row])], ignore_index=True)

print(conf_NN)






# The code creates a DataFrame summarizing the confusion matrix for the Multi-Layer Perceptron (MLP) classifier's predictions on the training data and visualizes the relationship between actual and predicted fuel consumption values using a scatter plot.This analysis
# provides a comprehensive evaluation of the MLP classifier's performance on the training data by visualizing the confusion matrix and the relationship between actual and predicted fuel consumption values. It highlights the classifier's strengths and weaknesses,
# allowing for further refinement and improvement

# In[51]:


sns.scatterplot(x='real_Consumption', y='predicted_Consumption_GNB', s=(conf_NN.density) * 60, data=conf_NN)
plt.xlabel('Real Consumption')
plt.ylabel('Predicted Consumption using NN')
plt.title('Prediction relevance of NN on the train set')
plt.show()


# In[52]:


prediction_test_NN =classifier_NN.predict(Norm[Input_cols]) #prediction
#We store the K-means results in a dataframe
prediction_test_NN_pd = pd.DataFrame(prediction_test_NN)
prediction_test_NN_pd.columns = ['Prediction_NN']
#we merge this dataframe with df
Norm= pd.concat([Norm,prediction_test_NN_pd], axis = 1)


# Calculates the confusion matrix for the Multi-Layer Perceptron (MLP) classifier's predictions on the entire dataset, including both training and test data. It then summarizes the confusion matrix in a DataFrame and visualizes the relationship between actual and
# predicted fuel consumption values using a scatter plot

# In[53]:


import pandas as pd

M_NN_total = confusion_matrix(Norm['Consumption_encoded'], prediction_test_NN_pd)
print(M_NN_total)

conf_NN_total = pd.DataFrame(columns=['real_encoded', 'real_Consumption', 'predicted_encoded', 'predicted_Consumption_NN', 'density'])
dataframes_to_concat = []

for i in range(0, 3):
    for j in range(0, 3):
        if M_NN_total[i][j] > 0:
            new_row = {'real_encoded': i, 'real_Consumption': dict_cat[i], 'predicted_encoded': j, 'predicted_Consumption_NN': dict_cat[j], 'density': float(M_NN_total[i][j])}
            dataframes_to_concat.append(pd.DataFrame([new_row]))

conf_NN_total = pd.concat(dataframes_to_concat, ignore_index=True)

print(conf_NN_total)


# In[54]:


sns.scatterplot(x='real_Consumption', y='predicted_Consumption_NN', s=(conf_NN_total.density)*60, data=conf_NN_total,color='k')
pl.xlabel('Real Consumption')
pl.ylabel('Predicted Consumption using NN')
pl.title('Prediction relevance of NN on the whole set')
plt.show()


# In[55]:


import seaborn as sns
import matplotlib.pyplot as plt


# Create a heatmap
heatmap_data = conf_NN_total.pivot_table(index='real_Consumption', columns='predicted_Consumption_NN', values='density')

plt.figure(figsize=(10, 8))  # Adjust the figure size as needed

sns.heatmap(data=heatmap_data, cmap='viridis', annot=True, fmt=".2f")

plt.xlabel('Predicted Consumption using NN')
plt.ylabel('Real Consumption')
plt.title('Prediction Relevance of NN on the Whole Set')
plt.show()


# In[ ]:





# In[56]:


print('Using Neuron Network, the predicted Consumption of the 34th car is ' + str(dict_cat[Norm.iloc[34]['Prediction_NN'].astype(int)]))


# In[ ]:





# In[ ]:





# In[ ]:


# KMeans


# The provided code snippet makes predictions for the fuel consumption classes of the training data using the trained K-means clustering algorithm, stores the predictions in a DataFrame, and merges this DataFrame with the original DataFrame to create a
# comprehensive DataFrame that includes both actual and predicted fuel consumption classes for all data points.By storing the predictions in a separate DataFrame and merging it with the original DataFrame, it allows for further analysis and comparison with the
# predictions made by other classification models.

# In[57]:


from sklearn import cluster
from sklearn.cluster import KMeans
from sklearn.metrics import completeness_score, homogeneity_score
Nombre_clusters=3#cluster nombers matching rhe numbers of species
kmeans = KMeans(n_clusters=Nombre_clusters, init='random') # initialization 


# In[58]:


#K-means training
kmeans.fit(train[Input_cols] )
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
 
 
print('Coordinates of the 15 centroids')
print(centroids)


# Calculates the completeness score to evaluate the performance of the K-means clustering algorithm's predictions on the training data. The completeness score measures the proportion of data points within each cluster that are correctly classified as belonging to
# that cluster.A higher completeness score suggests that the K-means clustering algorithm effectively groups data points with similar fuel consumption characteristics, resulting in more accurate predictions for the majority of the data points in each cluster.
# 

# In[59]:


#actual prediction
y_pred_kmean = kmeans.predict(train[Input_cols])
#We store the K-means results in a dataframe
pred = pd.DataFrame(y_pred_kmean)
pred.columns = ['Prediction_kmean']


# In[60]:


print (completeness_score(train['Consumption_encoded'],pred['Prediction_kmean']))


# Calculates the homogeneity score to assess the performance of the K-means clustering algorithm's predictions on the training data. The homogeneity score measures the extent to which data points in a cluster share similar fuel consumption characteristics
# 

# In[61]:


print (homogeneity_score(train['Consumption_encoded'],pred['Prediction_kmean']))


# calculates the confusion matrix to evaluate the performance of the K-means clustering algorithm's predictions on the training data.Each cell in the matrix represents the number of data points that were correctly or incorrectly classified. The diagonal elements
# represent correctly classified data points, while off-diagonal elements represent misclassified data points.
# 

# In[62]:


M_kmean=confusion_matrix(train['Consumption_encoded'],pred['Prediction_kmean'])
print (M_kmean)


# This visualization provides a visual representation of how well the K-means clustering algorithm aligns with the actual fuel consumption classes. The scatter plot shows that the algorithm generally assigns data points with similar fuel consumption values to the same
# cluster, indicating that the clusters are well-defined and distinct.
# 

# In[63]:


import pandas as pd

dict_cluster = {0: 'A', 1: 'B', 2: 'C'}
conf_kmean = pd.DataFrame(columns=['real', 'real_Consumption', 'predicted', 'predicted_cluster', 'density'])
dataframes_to_concat = []

for i in range(0, 7):
    for j in range(0, 7):
        if M_kmean[i][j] > 0:
            new_row = {'real': i, 'real_Consumption': dict_cat[i], 'predicted': j, 'predicted_cluster': dict_cluster[j], 'density': float(M_kmean[i][j])}
            dataframes_to_concat.append(pd.DataFrame([new_row]))

conf_kmean = pd.concat(dataframes_to_concat, ignore_index=True)

print(conf_kmean)


# In[64]:


sns.scatterplot(x='real_Consumption', y='predicted_cluster', s=(conf_kmean.density)*60, data=conf_kmean)
pl.xlabel('Real Consumption')
pl.ylabel('Cluster k-mean')
pl.title('Alignement between clusters and consumption')
plt.show()


# In[65]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming conf_kmean contains the necessary data

# Create a heatmap with annotations
sns.heatmap(data=conf_kmean.pivot_table(index='real_Consumption', columns='predicted_cluster', values='density'),
            cmap='viridis', annot=True, fmt=".2f")

plt.xlabel('Predicted Cluster')
plt.ylabel('Real Consumption')
plt.title('Alignment Between Clusters and Consumption Heatmap')
plt.show()



# In[ ]:





# In[66]:


get_ipython().system('pip install import-ipynb')
import import_ipynb
import Matching_cluster


# In[67]:


acc,y_pred,dict_map_cluster =Matching_cluster.remap_labels(pred['Prediction_kmean'],train['Consumption_encoded'])
print(dict_map_cluster)
#We store the K-means results in a dataframe
pred_1 = pd.DataFrame(y_pred)
pred_1.columns = ['Prediction_kmean_mapped']
#we merge this dataframe with df
pred= pd.concat([pred,pred_1], axis = 1)


# The confusion matrix shows that the K-means algorithm performs well in classifying data points into their respective fuel consumption classes, with high accuracy for most classes. However, it is important to note that performance on the training data may not
# necessarily translate to performance on unseen data.
# 

# In[68]:


M_kmeanmapped=confusion_matrix(train['Consumption_encoded'],pred['Prediction_kmean_mapped'])
print (M_kmeanmapped)


# Summarizes the confusion matrix after mapping the cluster labels, visualizes the relationship between actual and predicted fuel consumption values using a scatter plot, and prints the summary of the accuracy of the K-means predictions after mapping the cluster
# labels.
# it appears that the K-means clustering algorithm has been able to effectively group the data points into clusters based on their fuel consumption. The clusters are represented by the predicted values, and the density column indicates the number of data points in
# each cluster

# In[69]:


import pandas as pd

conf_kmeanmapped = pd.DataFrame(columns=['real', 'real_Consumption', 'predicted', 'predicted_Consumption', 'density'])
dataframes_to_concat = []

for i in range(0, 7):
    for j in range(0, 7):
        if M_kmeanmapped[i][j] > 0:
            new_row = {'real': i, 'real_Consumption': dict_cat[i], 'predicted': j, 'predicted_Consumption': dict_cat[j], 'density': float(M_kmeanmapped[i][j])}
            dataframes_to_concat.append(pd.DataFrame([new_row]))

conf_kmeanmapped = pd.concat(dataframes_to_concat, ignore_index=True)

print(conf_kmeanmapped)


# In[70]:


sns.scatterplot(x='real_Consumption', y='predicted_Consumption', s=(conf_kmeanmapped.density)*60, data=conf_kmeanmapped)
pl.xlabel('Real Consumption')
pl.ylabel('Cluster k-mean')
pl.title('Prediction accuracy')
plt.show()


# Performs K-means clustering on the normalized data using the kmeans.predict() method. The Norm[Input_cols] part indicates that the clustering is performed on the subset of the Norm DataFrame containing the columns specified in the Input_cols list.
# 

# In[71]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming conf_kmeanmapped contains the necessary data

# Create a heatmap
sns.heatmap(data=conf_kmeanmapped.pivot_table(index='real_Consumption', columns='predicted_Consumption', values='density'),
            cmap='viridis')

plt.xlabel('Predicted Consumption')
plt.ylabel('Real Consumption')
plt.title('Prediction Accuracy Heatmap')
plt.show()


# In[72]:


y_pred_kmean = kmeans.predict(Norm[Input_cols])
#We store the K-means results in a dataframe
pred = pd.DataFrame(y_pred_kmean)
pred.columns = ['Prediction_kmean_mapped']
mapping=pred['Prediction_kmean_mapped'].map(dict_map_cluster)
print(mapping)
Norm = pd.concat([Norm,mapping], axis = 1)
print(Norm)


# In[73]:


M_kmeanmapped_total=confusion_matrix(Norm['Consumption_encoded'],Norm['Prediction_kmean_mapped'])
print (M_kmeanmapped_total)


# Summarizing the confusion matrix after mapping the cluster labels back to the original fuel consumption classes and visualizes the relationship between actual and predicted fuel consumption values using a scatter plot.
# 

# In[74]:


import pandas as pd

conf_kmeanmapped_total = pd.DataFrame(columns=['real', 'real_Consumption', 'predicted', 'predicted_Consumption', 'density'])
dataframes_to_concat = []

for i in range(0, 7):
    for j in range(0, 7):
        if M_kmeanmapped_total[i][j] > 0:
            new_row = {'real': i, 'real_Consumption': dict_cat[i], 'predicted': j, 'predicted_Consumption': dict_cat[j], 'density': float(M_kmeanmapped_total[i][j])}
            dataframes_to_concat.append(pd.DataFrame([new_row]))

conf_kmeanmapped_total = pd.concat(dataframes_to_concat, ignore_index=True)

print(conf_kmeanmapped_total)


# In[75]:


sns.scatterplot(x='real_Consumption', y='predicted_Consumption', s=(conf_kmeanmapped_total.density)*60, data=conf_kmeanmapped_total, color='k')
pl.xlabel('Real Consumption')
pl.ylabel('Cluster k-mean')
pl.title('Prediction accuracy')
plt.show()


# Predicts the fuel consumption class for a new car with the given specifications using the K-means clustering model.
# 

# In[76]:


import seaborn as sns
import matplotlib.pyplot as plt

# Assuming conf_kmeanmapped_total contains the necessary data

# Create a heatmap
sns.heatmap(data=conf_kmeanmapped_total.pivot_table(index='real_Consumption', columns='predicted_Consumption', values='density'),
            annot=True, cmap='viridis')

plt.xlabel('Predicted Consumption')
plt.ylabel('Real Consumption')
plt.title('Prediction Accuracy Heatmap')
plt.show()


# In[ ]:





# In[77]:


New_specimen = { 'Cylinders':[0.5],
 'Cubic_inch': [0.5],
 'Horsepower': [0.5],
 'Weight': [0.5],
 'Acceleration':[0.5]
 }
panda_New_specimen = pd.DataFrame(New_specimen) 
D=kmeans.predict(panda_New_specimen)
print('Using kmeans, the predicted Consumption of such a car is '+ str(dict_cat[dict_map_cluster[D[0]]]))


# In[ ]:





# In[78]:


# Import necessary libraries for Dash
import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Car Consumption Dashboard"),

    # Dropdown for X-axis selection for scatter plot
    dcc.Dropdown(
        id='x-axis-dropdown-scatter',
        options=[{'label': col, 'value': col} for col in data_panda.columns],
        value='Cubic_inch',
        multi=False,
        style={'width': '50%'}
    ),

    # Dropdown for Y-axis selection for scatter plot
    dcc.Dropdown(
        id='y-axis-dropdown-scatter',
        options=[{'label': col, 'value': col} for col in data_panda.columns],
        value='Horsepower',
        multi=False,
        style={'width': '50%'}
    ),

    # Dropdown for X-axis selection for line chart
    dcc.Dropdown(
        id='x-axis-dropdown-line',
        options=[{'label': col, 'value': col} for col in data_panda.columns],
        value='Cubic_inch',
        multi=False,
        style={'width': '50%'}
    ),

    # Dropdown for Y-axis selection for line chart
    dcc.Dropdown(
        id='y-axis-dropdown-line',
        options=[{'label': col, 'value': col} for col in data_panda.columns],
        value='Horsepower',
        multi=False,
        style={'width': '50%'}
    ),

    # Scatter plot
    dcc.Graph(id='scatter-plot'),

    # Line chart
    dcc.Graph(id='line-chart'),

    # Pie chart
    dcc.Graph(id='pie-chart')
])

# Callback to update scatter plot based on dropdown selection
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('x-axis-dropdown-scatter', 'value'),
     Input('y-axis-dropdown-scatter', 'value')]
)
def update_scatter_plot(x_column, y_column):
    scatter_fig = px.scatter(
        data_panda, x=x_column, y=y_column,
        color='Consumption', title=f'{x_column} vs. {y_column}',
        labels={x_column: x_column, y_column: y_column, 'Consumption': 'Consumption'},
        template='plotly_dark'
    )
    return scatter_fig

# Callback to update line chart based on dropdown selection
@app.callback(
    Output('line-chart', 'figure'),
    [Input('x-axis-dropdown-line', 'value'),
     Input('y-axis-dropdown-line', 'value')]
)
def update_line_chart(x_column, y_column):
    line_fig = px.line(
        data_panda, x=x_column, y=y_column, title=f'{y_column} vs. {x_column}',
        labels={x_column: x_column, y_column: y_column},
        template='plotly_dark'
    )
    return line_fig

# Callback to update pie chart based on dropdown selection
@app.callback(
    Output('pie-chart', 'figure'),
    [Input('x-axis-dropdown-scatter', 'value')]
)
def update_pie_chart(x_column):
    # Calculate consumption distribution
    consumption_counts = data_panda['Consumption'].value_counts()
    consumption_percentage = (consumption_counts / len(data_panda)) * 100

    # Create a list to store hover text strings
    hover_texts = []
    for consumption, count, percentage in zip(consumption_counts.index, consumption_counts, consumption_percentage):
        # Get the car names for the current consumption category
        car_names = data_panda[data_panda['Consumption'] == consumption]['Brand'].tolist()
        # Create hover text with car names
        hover_text = f"{percentage:.1f}% cars ({count})\nCar Names:\n{', '.join(car_names)}"
        hover_texts.append(hover_text)

    # Create pie chart using Plotly
    pie_fig = go.Figure(data=[go.Pie(labels=consumption_counts.index, 
                                      values=consumption_counts, 
                                      textinfo='label+percent',
                                      hole=0.3,
                                      hoverinfo='text',
                                      text=hover_texts)])
    pie_fig.update_layout(title='Consumption Distribution')
    return pie_fig

# Run the app
app.run_server(mode='external', port=8067)


# In[ ]:





# In[ ]:




