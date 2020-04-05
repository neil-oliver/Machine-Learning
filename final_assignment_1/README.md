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

## Iteration 2
A lot of different approaches were tested out with little to no improvement on the training or test results. 
Testing included a combination of the following changes:
### Count Vectorizer
- ```ngram_range``` using tuples (1,2),(1,3),(1,4),(1,5),(2,2),(2,3),(2,5),(3,5). 
Performance decreased when moving away from the lower limit of 1.  

- ```lowercase`` True & False. 
Performance decreased on False  

- ```max_df``` 0.1, 0.3, 0.5, 0.8. Performance decreased on lower values. 
Optimal was 0.5, all other values spread out the performance from the different models.  

- ```min_df``` 5, 10, 25 ( words appearing in either 10 or 25 reviews), 0.01, 0.02 (words appearing in 1% or 2% of reviews). 
Percentage values made more sense, but the fixed values showed a clearer affect on results. Possibly the perfentage value was too high. Optimal was 5.  

- custom preprocessor to reduce noise in the reviews

### Custom Features
- Removal of all custom features from the first iteration except word and sentence count ("." & " ").
The removal of these features surprisingly had very little affect to the training or test results.

### Model Optimization
As both the SVM and Logistic Regression models had both performed in a very similar manner in the first iteration (with only 0.5% difference on test results) and during the changes to the features in the second iteration, model optimization was carried out on both models. A general observation on both was they seem to be overfitting on the test data, so a focus was on trying to reduce the overfitting.

- ```alpha``` 10, 5, 4, 3, 2, 1,0, 1,0.01, 0.001, 0.0001, 0.00001.
Some feature selection gave polar results, with either an extrmemly high false negative or false positive. Additional alpha tuning was carried out using a binary search with no success.  

- ```alpha``` 100000, 10000, 1000, 100, 10.
No positive results from limiting the number of itterations.  

- ```penalty``` l2, l1, elasticnet.
These affect the results a lot and each of the alpha values needed to be tested with each different penalty. The elasticnet showed the biggest difference where the models did not seem to overfit as much. I was hopeful that this would allow further optimization of the alpha values and other paramters but there was still no success in moving the accuracy or precision of the test results above 90%. While the results were similar, the default ```l2``` still performanced consistently better by a minimal amount (0.1 - 0.2%).

- ```early_stopping```True, False.
Setting the early stopping parameter to False did have a big affect on the training results and i believe helped stop overfitting. The results marginally improved the test results but also by a minimal amount (0.1 - 0.2%). Additional tweaking of ```n_iter_no_change``` needs to be explored in future iterations.

### Model Selection
While both models use the ```sklearn.linear_model.SGDClassifier``` with different loss models. The ```log``` loss performed slightly better on all tests, reaching for the first time over 90% (90.02%). This is only a marginal improvement from the first itteration result of 89.7% but still a consistent improvement. The submitted model therefore uses the ```log``` loss function to see if it generalises better to the submission test data.



