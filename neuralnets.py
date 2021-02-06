from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

data = pd.read_csv("amazon_dataframe.csv")

train_data = data.head(40000)
test_data = data.tail(10000)

train_data  = train_data [train_data ['star_rating'] != 3]
test_data  = test_data [test_data ['star_rating'] != 3]

# if star_rating is above 3, sentiment is positive ; if star_rating is below 3, sentiment is negative
train_data ['feedback sentiment'] = train_data ['star_rating'].apply(lambda rating : 1 if rating > 3.0 else 0 )
test_data ['feedback sentiment'] = test_data ['star_rating'].apply(lambda rating : 1 if rating > 3.0 else 0 )

print(train_data)
print(test_data)

tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')

train_matrix = tokenizer.fit_transform(train_data['review_body'].values.astype('U'))
test_matrix = tokenizer.transform(test_data['review_body'].values.astype('U'))

clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf.fit(train_matrix.todense(),train_data['feedback sentiment'])

prediction = clf.predict(test_matrix.todense())

print(confusion_matrix(test_data['feedback sentiment'],prediction))
print(classification_report(test_data['feedback sentiment'],prediction))
