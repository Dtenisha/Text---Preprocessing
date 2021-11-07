#!/usr/bin/env python
# coding: utf-8

# <h2> Natural Language Processing <h2>

# In[1]:


#1.1 Text Preprocessing
import nltk
nltk.download


# In[2]:


text1= 'Text Mining is the process of EXTRACTING or RETRIEVING Information from large text@data???.Text """" preprocessing is the first @step?!!! in text mining'


# In[3]:


### lowercasing 
text2=text1.lower()
text2


# In[4]:


### removing whitespace
text3=text2.strip()
text3


# In[5]:


### removing punctuation
import string
print(string.punctuation)


# In[6]:


remove_punctuation="".join([char for char in text3 if char not in string.punctuation])
remove_punctuation


# In[7]:


##tokenization
from nltk import word_tokenize
tokenize_words=word_tokenize(remove_punctuation)
tokenize_words


# In[8]:


### word filtering
from nltk.corpus import stopwords
stop_words=stopwords.words('english')
print(stop_words)


# In[9]:


filtered_words = [char for char in tokenize_words if char not in stop_words]
filtered_words


# In[10]:


### stemming
from nltk.stem.porter import PorterStemmer
porter=PorterStemmer()
stemmed=[porter.stem(word) for word in filtered_words]
print(stemmed)


# In[11]:


### Lemmatization
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
lemma_words=[]
for word in filtered_words:
    lemma_words.append(lemmatizer.lemmatize(word))
    print(lemmatizer.lemmatize(word))


# In[12]:


### POS Tagger
nltk.download('averaged_perceptron_tagger')
from nltk import pos_tag
poss=pos_tag(filtered_words)
print(poss)


# In[ ]:





# <h3>1.2 Bag Of Words <3>

# In[13]:


import sklearn 
from sklearn.feature_extraction.text import CountVectorizer


# In[14]:


phrases=["Bag of Words is a method to extract features from text documents.",
         "These features can be used for training machine learning algorithms.",
         "It creates a vocabulary of all the unique words occurring in all the documents in the training set."]
phrases


# In[15]:


count_vect=CountVectorizer()
count_vect.fit(phrases)


# In[16]:


print("vocabulary size:{}".format(len(count_vect.vocabulary_)))
print("vocabulary content:{}".format(count_vect.vocabulary_))


# In[17]:


bag_of_words=count_vect.transform(phrases)
print(bag_of_words)


# In[18]:


##converting bag of words to an array

print("bag_of_words as an array:\n{}".format(bag_of_words.toarray()))


# In[19]:


bag_of_words=bag_of_words.toarray()
vocabulary=count_vect.vocabulary_


# In[20]:


import pandas as pd


# In[21]:


df=pd.DataFrame(bag_of_words,columns=vocabulary)
df


# In[ ]:





# <h3>1.3 N gram <h3>

# In[22]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


# In[23]:


n_grams=CountVectorizer(ngram_range=(2,4))


# In[24]:


text=["Text mining can be broadly defined as a knowledge-intensive process." 
      "in which a user interacts with a document collection over time by using various types of analysis tools in Text mining.",
     "Text can be both structured and unstructured"]
text


# In[25]:


x=n_grams.fit_transform(text)
print(n_grams.get_feature_names())
x=n_grams.transform(text)
print(x.toarray())


# In[26]:


pd.set_option("display.max_columns",None)
df=pd.DataFrame(x.toarray(),columns=n_grams.get_feature_names())
df


# <h3> 1.4 Vector Space Models <h3>

# In[27]:


from sklearn .feature_extraction.text import TfidfVectorizer


# In[28]:


text_data=["Text preprocessing Tasks include all those routines, processes, and methods required to prepare data for a text mining.",
"Text Analytics is the process of drawing meaning out of written communication.",
"In a customer experience context, text analytics means examining text that was written by, or about, customers."]
text_data


# In[29]:


vectorizer=TfidfVectorizer()
x=vectorizer.fit_transform(text_data)
print(vectorizer.get_feature_names())
print(x.shape)


# In[30]:


import pandas as pd
vector=x
df1=pd.DataFrame(vector.toarray(),columns=vectorizer.get_feature_names())
df1

