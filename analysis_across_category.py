import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

def counter(arr):
    count = 0
    for i in arr:
        # Check if i is indexable (e.g. a list or array), otherwise compare directly.
        if hasattr(i, '__getitem__'):
            if i[0] < 0.5:
                count += 1
        else:
            if i < 0.5:
                count += 1
    return count

def counter2(arr):
    return sum(1 for i in arr if i == 1)

def main():
    # Read data once
    data = pd.read_csv("Data/office_products.csv")
    
    # Split data: first 80k for training, last 20k for testing
    train_data = data.head(80000)
    test_data = data.tail(20000)
    
    # Remove rows with ambiguous star_rating == 3
    train_data = train_data[train_data['star_rating'] != 3]
    test_data = test_data[test_data['star_rating'] != 3]
    
    # Define feedback sentiment: 1 if star_rating > 3, else -1
    train_data['feedback sentiment'] = train_data['star_rating'].apply(lambda r: 1 if r > 3.0 else -1)
    test_data['feedback sentiment'] = test_data['star_rating'].apply(lambda r: 1 if r > 3.0 else -1)
    
    print("Training data sample:")
    print(train_data.head())
    print("\nTesting data sample:")
    print(test_data.head())
    
    # Initialize CountVectorizer with a simple word pattern
    tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')
    
    # Create document-term matrices from the review text
    train_matrix = tokenizer.fit_transform(train_data['review_body'].astype(str).values)
    test_matrix = tokenizer.transform(test_data['review_body'].astype(str).values)
    
    # Initialize and train the Logistic Regression model
    logit = LogisticRegression(solver='liblinear')
    logit.fit(train_matrix, train_data['feedback sentiment'])
    
    # Output model intercept and vocabulary details
    print("\nIntercept:", logit.intercept_)
    word_arr = tokenizer.get_feature_names_out()
    print("Vocabulary size:", len(word_arr))
    
    # Create a DataFrame for word coefficients
    coef_list = logit.coef_[0].tolist()
    coef_df = pd.DataFrame({"Word": list(word_arr), "Coef": coef_list})
    print("\nWord Coefficients (sample):")
    print(coef_df.head())
    
    # Filter for words with coefficients greater than 2 or less than -2
    significant_coef_df = coef_df[(coef_df['Coef'] > 2) | (coef_df['Coef'] < -2)]
    significant_coef_df.to_csv("word_coef_across_category.csv", index=False, encoding="utf-8")
    
    # Evaluation and output
    print("\nDECISION FUNCTION:")
    decision_values = logit.decision_function(test_matrix)
    print(decision_values)
    
    print("\nPROBABILITY:")
    print("Classes:", logit.classes_)
    print(logit.predict_proba(test_matrix))
    
    print("\nPREDICTION:")
    predictions = logit.predict(test_matrix)
    print(predictions)
    
    print("\nCounter of decision function probabilities (first component < 0.5):", 
          counter(logit.predict_proba(test_matrix)))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_data['feedback sentiment'], predictions))
    
    print("\nClassification Report:")
    print(classification_report(test_data['feedback sentiment'], predictions))

if __name__ == "__main__":
    main()
