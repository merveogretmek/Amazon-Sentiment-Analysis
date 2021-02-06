from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix,classification_report
import pandas as pd
import numpy as np

data = pd.read_csv("amazon_dataframe.csv")

data = data.tail(10000)

print(data)

# remove the row if star_rating = 3 (cannot interpret 3 as positive or negative)
data  = data [data ['star_rating'] != 3]

# create a new column "feedback sentiment"
# if star_rating is above 3, sentiment is positive ; if star_rating is below 3, sentiment is negative
data ['feedback sentiment'] = data ['star_rating'].apply(lambda rating : 1 if rating > 3.0 else -1 )

print(data)


def sentiment_scores(sentence):
    # Create a SentimentIntensityAnalyzer object.
    sid_obj = SentimentIntensityAnalyzer()

    # polarity_scores method of SentimentIntensityAnalyzer
    # oject gives a sentiment dictionary.
    # which contains pos, neg, neu, and compound scores.
    sentiment_dict = sid_obj.polarity_scores(sentence)

    # decide sentiment as positive, negative and neutral
    if sentiment_dict['compound'] >= 0.05:
        return 1

    else :
        return -1

    # Driver code


data["sentiment"] = np.nan

for i in data.index:
    data.loc[i,"sentiment"] = sentiment_scores(str(data["review_body"][i]))

print(data)

print(confusion_matrix(data["feedback sentiment"].astype(int),data["sentiment"]))
print(classification_report(data["feedback sentiment"].astype(int),data["sentiment"]))

