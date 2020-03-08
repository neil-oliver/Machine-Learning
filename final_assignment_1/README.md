# Machine Learning Final Assignment 1 

### Topic Selection: Movie Reviews

### Model Selection: SVM

## Major Changes to Sample Code

### AFINN sentiment dictionary.
Each review was given a score based on the total value of words that matched the AFINN dictionary. The python AFINN module was used to handle the totalling and additional characters / basic lemmatizing before scoring. The dictionary was modified to remove all positive words and convert negative scores into positive scores. This score was added to the quantitative features array.

```python
from afinn import Afinn
afn = Afinn()

movie_data['afinn_score'] = [afn.score(c) for c in movie_data['review']]
```
[AFINN Python Github](https://github.com/fnielsen/afinn)

### Additional quantitative features
Count figures were added for additional symbols including **?**, **!**, **#** & **\***. A count of upper case letters to detect if people **MIGHT BE SHOUTING** was also included.

```python
movie_data['q_count'] = movie_data['review'].str.count("\?")
movie_data['q_star'] = movie_data['review'].str.count("\*")
movie_data['q_hash'] = movie_data['review'].str.count("\#")

import string
movie_data['upper'] = [sum(1 for letter in c if letter.isupper()) for c in movie_data['review']]
```
### Lemmatizer
A lemmatizer function was added as discussed in the [Scikit Learn Documentation](https://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes).
The lemmatizer aims to combine words that are the same (hate, hates etc) during the countVectorizer stage. Stop words were also lemmatized for use in the vecotizer and ignored in the lemmatize class.

```python
import nltk
from nltk import word_tokenize 
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer 

nltk.download('stopwords', quiet=True, raise_on_error=True)
stop_words = set(nltk.corpus.stopwords.words('english'))
tokenized_stop_words = nltk.word_tokenize(' '.join(nltk.corpus.stopwords.words('english')))

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, articles):
        return [self.wnl.lemmatize(t) for t in word_tokenize(articles) if t not in stop_words]
```

### Hashing Vectorizer to CountVectorizer
Performance improvements were found when using the CountVectorizer over the HashingVectorizer, especally in combination with the Lemmatizer function.
Stop words were removed, accents stripped, words converted to lowercase & n-grams included. 

```python
hv = CountVectorizer(tokenizer=LemmaTokenizer(), strip_accents='ascii', ngram_range=(1,2), lowercase=True, max_df = 0.5, min_df = 10, stop_words=tokenized_stop_words)
```

### SVM Model Settings
Alpha Values and maximum iterations were adjusted to **lower** the training result accuracy as the model was overfitting by default. Alpha values were tested across all SGDClassifier models to ensure consistency and generalisability. 

```python
from sklearn import linear_model
svm = linear_model.SGDClassifier(alpha=5, max_iter=10000)
svm.fit(X_train, y_train)

svm_performance_train = BinaryClassificationPerformance(svm.predict(X_train), y_train, 'svm_train')
svm_performance_train.compute_measures()
print(svm_performance_train.performance_measures)
```