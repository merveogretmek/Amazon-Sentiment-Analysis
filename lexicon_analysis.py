from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

def main():
    # Load the data and select the last 10,000 rows
    data = pd.read_csv("amazon_dataframe.csv").tail(10000)
    
    # Remove rows with star_rating == 3 (ambiguous)
    data = data[data['star_rating'] != 3]
    
    # Create the "feedback sentiment" column:
    # Positive (1) if star_rating > 3, Negative (-1) otherwise.
    data['feedback sentiment'] = data['star_rating'].apply(lambda r: 1 if r > 3.0 else -1)
    
    # Create a single SentimentIntensityAnalyzer instance
    analyzer = SentimentIntensityAnalyzer()
    
    # Compute sentiment scores for each review using vectorized apply
    data["sentiment"] = data["review_body"].astype(str).apply(
        lambda sentence: 1 if analyzer.polarity_scores(sentence)['compound'] >= 0.05 else -1
    )
    
    # Print evaluation metrics
    print(confusion_matrix(data["feedback sentiment"].astype(int), data["sentiment"]))
    print(classification_report(data["feedback sentiment"].astype(int), data["sentiment"]))

if __name__ == "__main__":
    main()
