#!/usr/bin/env python
# coding: utf-8


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from wordcloud import WordCloud


import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords

data=pd.read_csv('AmazonReview.csv')
data.head()
data.info()

# Now, To drop the null values (if any), run the below command.
data.dropna(inplace=True)

# To predict the Sentiment as positive(numerical value = 1) or negative(numerical value = 0), we need to change them the values to those categories. For that the condition will be like if the sentiment value is less than or equal to 3, then it is negative(0) else positive(1). For better understanding, refer the code below.
#1,2,3->negative(i.e 0)
data.loc[data['Sentiment']<=3,'Sentiment'] = 0

#4,5->positive(i.e 1)
data.loc[data['Sentiment']>3,'Sentiment'] = 1 

# Now, once the dataset is ready, we will clean the review column by removing the stopwords. The code for that is given below.
stp_words=stopwords.words('english')
def clean_review(review): 
  cleanreview=" ".join(word for word in review.
                       split() if word not in stp_words)
  return cleanreview 

data['Review']=data['Review'].apply(clean_review)


# Once we have done with the preprocess. Let’s see the top 5 rows to see the improved dataset.

data.head()


# ## Analysis of the Dataset
# Let’s check out that how many counts are there for positive and negative sentiments.

data['Sentiment'].value_counts()


# To have the better picture of the importance of the words let’s create the Wordcloud of all the words with sentiment = 0 i.e. negative
consolidated=' '.join(word for word in data['Review'][data['Sentiment']==0].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# Let’s do the same for all the words with sentiment = 1 i.e. positive

consolidated=' '.join(word for word in data['Review'][data['Sentiment']==1].astype(str))
wordCloud=WordCloud(width=1600,height=800,random_state=21,max_font_size=110)
plt.figure(figsize=(15,10))
plt.imshow(wordCloud.generate(consolidated),interpolation='bilinear')
plt.axis('off')
plt.show()


# Now we have a clear picture of the words we have in both the categories.
# Let’s create the vectors.
# 
# ### Converting text into Vectors
# TF-IDF calculates that how relevant a word in a series or corpus is to a text. The meaning increases proportionally to the number of times in the text a word appears but is compensated by the word frequency in the corpus (data-set). We will be implementing this with the code below.


cv = TfidfVectorizer(max_features=2500)
X = cv.fit_transform(data['Review'] ).toarray()


# ### Model training, Evaluation, and Prediction
# Once analysis and vectorization is done. We can now explore any machine learning model to train the data. But before that perform the train-test split.


from sklearn.model_selection import train_test_split
x_train ,x_test,y_train,y_test=train_test_split(X,data['Sentiment'],
                                                test_size=0.25 ,
                                                random_state=42)


# Now we can train any model, Let's explore the Logistic Regression

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model=LogisticRegression()

#Model fitting
model.fit(x_train,y_train)

#testing the model
pred=model.predict(x_test)

#model accuracy
print(accuracy_score(y_test,pred))

# Will check confusion matrix
from sklearn import metrics
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)
cm_display=metrics.ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=[False,True])
cm_display.plot()
plt.show()
