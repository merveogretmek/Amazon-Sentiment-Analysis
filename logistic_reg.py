import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

def counter(arr):
    count = 0
    for i in arr:
        if i[0] < 0.5:
            count = count + 1
    return count

def counter2(arr):
    count = 0
    for i in arr:
        if i == 1:
            count = count + 1
    return count

# read data from csv
dataframe = pd.read_csv("amazon_dataframe.csv")

train_data = dataframe.head(40000)
test_data = dataframe.tail(10000)

train_data  = train_data [train_data ['star_rating'] != 3]
test_data  = test_data [test_data ['star_rating'] != 3]

# if star_rating is above 3, sentiment is positive ; if star_rating is below 3, sentiment is negative
train_data ['feedback sentiment'] = train_data ['star_rating'].apply(lambda rating : 1 if rating > 3.0 else -1 )
test_data ['feedback sentiment'] = test_data ['star_rating'].apply(lambda rating : 1 if rating > 3.0 else -1 )

# 80% of the observations will be used for training (240000/300000)
train = train_data
print(train)

# 20% of the observations will be used for training (60000/300000)
test = test_data
print(test)

# regular expression '\b\w+\b' will capture the words from the text by escaping whitespace
# create a vocabulary pool from the observed text
tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')


# create train_matrix
# astype(U) : convert numpy array to Unicode (feedback text might contain many different characters)
train_matrix = tokenizer.fit_transform(train['review_body'].values.astype('U'))
#print(train_matrix)
# first 5 line from printing train_matrix :
#  (0, 1383)	1
#  (0, 34257)	1
#  (0, 43930)	1
#  (0, 31836)	3
#  (0, 64184)	1

# Observe the first row:
# 0 is the sentence index from tokenizer.vocabulary_
# 1383 the word index from tokenizer.vocabulary_
# 1 is the number of times the word(with index number = 1383) appears in this feedback text



#create test_matrix
# 5: review body
test_matrix = tokenizer.transform(test['review_body'].values.astype('U'))

# define a Logistic Regression Model: classifier
logit = LogisticRegression(solver = 'liblinear')


# fit the model with given training data
logit.fit(train_matrix,train['feedback sentiment'])
#print(counter(logit.decision_function(test_matrix)))
print(logit.intercept_)
word_arr = tokenizer.get_feature_names()
print(word_arr)
print(len(word_arr))

word_list = []
for i in word_arr:
    word_list.append(i)


coef_arr = logit.coef_

coef_list = coef_arr[0].tolist()

df2 = pd.DataFrame({"Word":word_list,
                   "Coef":coef_list})

print(df2)

df2  = df2 [(df2 ['Coef']> 2) | (df2 ['Coef'] < -2)]



df2.to_csv("word_coef_df.csv", index=False, encoding="utf-8")


print("\n")
print("\n")
print("DECISION FUNCTION")
print(logit.decision_function(test_matrix))

print("PROBABILITY")
print(logit.classes_)
print(logit.predict_proba(test_matrix))
print("\n")
print("PREDICTION")
print(logit.predict(test_matrix))
print("\n")
print(counter(logit.predict_proba(test_matrix)))
print("\n")
#print(f"Range of Decision Function: min = {min(logit.decision_function(test_matrix))}, max = {max(logit.decision_function(test_matrix))}")

print("\n")
print("\n")



# predict the sentiment for observations in testing data
prediction = logit.predict(test_matrix)
#print(counter2(prediction))

print(confusion_matrix(test['feedback sentiment'],prediction))
print(classification_report(test['feedback sentiment'],prediction))



