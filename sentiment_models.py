import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer

# ----------------------------
# Experiment 1: Decision Tree
# ----------------------------
def experiment_decision_tree(data):
    from sklearn.tree import DecisionTreeClassifier

    print("\n=== Decision Tree Experiment ===")
    df = data.copy()
    # Use 80k for training, 20k for testing; remove rating == 3
    train_data = df.head(80000)
    test_data = df.tail(20000)
    train_data = train_data[train_data['star_rating'] != 3]
    test_data = test_data[test_data['star_rating'] != 3]

    # Define sentiment: 1 if star_rating > 3, else 0
    train_data['feedback sentiment'] = train_data['star_rating'].apply(lambda r: 1 if r > 3.0 else 0)
    test_data['feedback sentiment'] = test_data['star_rating'].apply(lambda r: 1 if r > 3.0 else 0)

    print("Train sample:")
    print(train_data.head())
    print("Test sample:")
    print(test_data.head())

    tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = tokenizer.fit_transform(train_data['review_body'].astype(str).values)
    test_matrix = tokenizer.transform(test_data['review_body'].astype(str).values)

    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(train_matrix, train_data['feedback sentiment'])
    prediction = dec_tree.predict(test_matrix)

    print("Confusion Matrix:")
    print(confusion_matrix(test_data['feedback sentiment'], prediction))
    print("Classification Report:")
    print(classification_report(test_data['feedback sentiment'], prediction))


# ----------------------------
# Experiment 2: K-Nearest Neighbors
# ----------------------------
def experiment_knn(data):
    from sklearn.neighbors import KNeighborsClassifier

    print("\n=== K-Nearest Neighbors Experiment ===")
    df = data.copy()
    # Use 40k for training, 10k for testing.
    train_data = df.head(40000)
    test_data = df.tail(10000)

    # Define sentiment: 1 if >3, -1 if <3, else 0
    train_data['feedback sentiment'] = train_data['star_rating'].apply(lambda r: 1 if r > 3.0 else (-1 if r < 3.0 else 0))
    test_data['feedback sentiment'] = test_data['star_rating'].apply(lambda r: 1 if r > 3.0 else (-1 if r < 3.0 else 0))

    print("Train sample:")
    print(train_data.head())
    print("Test sample:")
    print(test_data.head())

    tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = tokenizer.fit_transform(train_data['review_body'].astype(str).values)
    test_matrix = tokenizer.transform(test_data['review_body'].astype(str).values)

    neigh = KNeighborsClassifier(n_neighbors=3)
    neigh.fit(train_matrix, train_data['feedback sentiment'])
    prediction = neigh.predict(test_matrix)

    print("Confusion Matrix:")
    print(confusion_matrix(test_data['feedback sentiment'], prediction))
    print("Classification Report:")
    print(classification_report(test_data['feedback sentiment'], prediction))


# ----------------------------
# Experiment 3: Logistic Regression
# ----------------------------
def experiment_logistic_regression(data):
    from sklearn.linear_model import LogisticRegression

    def counter(arr):
        count = 0
        for i in arr:
            if i[0] < 0.5:
                count += 1
        return count

    print("\n=== Logistic Regression Experiment ===")
    df = data.copy()
    # Use 40k for training, 10k for testing; remove star_rating == 3.
    train_data = df.head(40000)
    test_data = df.tail(10000)
    train_data = train_data[train_data['star_rating'] != 3]
    test_data = test_data[test_data['star_rating'] != 3]

    # Define sentiment: 1 if >3 else -1.
    train_data['feedback sentiment'] = train_data['star_rating'].apply(lambda r: 1 if r > 3.0 else -1)
    test_data['feedback sentiment'] = test_data['star_rating'].apply(lambda r: 1 if r > 3.0 else -1)

    print("Train sample:")
    print(train_data.head())
    print("Test sample:")
    print(test_data.head())

    tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = tokenizer.fit_transform(train_data['review_body'].astype(str).values)
    test_matrix = tokenizer.transform(test_data['review_body'].astype(str).values)

    logit = LogisticRegression(solver='liblinear')
    logit.fit(train_matrix, train_data['feedback sentiment'])

    print("Logistic Regression Intercept:")
    print(logit.intercept_)
    # Use get_feature_names_out() for newer versions of scikit-learn.
    word_arr = tokenizer.get_feature_names_out()
    print("Vocabulary:")
    print(word_arr)
    print("Vocabulary size:", len(word_arr))

    coef_list = logit.coef_[0].tolist()
    coef_df = pd.DataFrame({"Word": list(word_arr), "Coef": coef_list})
    print("Coefficient DataFrame (full):")
    print(coef_df.head())
    # Filter coefficients larger than 2 or smaller than -2.
    coef_df_filtered = coef_df[(coef_df['Coef'] > 2) | (coef_df['Coef'] < -2)]
    coef_df_filtered.to_csv("word_coef_df.csv", index=False, encoding="utf-8")

    print("\nDECISION FUNCTION:")
    print(logit.decision_function(test_matrix))
    print("CLASS PROBABILITIES:")
    print(logit.classes_)
    print(logit.predict_proba(test_matrix))
    print("\nPREDICTIONS:")
    print(logit.predict(test_matrix))
    print("\nCounter (decision function):", counter(logit.predict_proba(test_matrix)))

    prediction = logit.predict(test_matrix)
    print("\nConfusion Matrix:")
    print(confusion_matrix(test_data['feedback sentiment'], prediction))
    print("Classification Report:")
    print(classification_report(test_data['feedback sentiment'], prediction))


# ----------------------------
# Experiment 4: One-vs-One SVM (LinearSVC)
# ----------------------------
def experiment_one_vs_one_svm(data):
    from sklearn.multiclass import OneVsOneClassifier
    from sklearn.svm import LinearSVC

    print("\n=== One-vs-One SVM Experiment ===")
    df = data.copy()
    # Use 40k for training, 10k for testing.
    train_data = df.head(40000)
    test_data = df.tail(10000)
    train_data['feedback sentiment'] = train_data['star_rating'].apply(
        lambda r: 1 if r > 3.0 else (-1 if r < 3.0 else 0))
    test_data['feedback sentiment'] = test_data['star_rating'].apply(
        lambda r: 1 if r > 3.0 else (-1 if r < 3.0 else 0))

    print("Train sample:")
    print(train_data.head())
    print("Test sample:")
    print(test_data.head())

    tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = tokenizer.fit_transform(train_data['review_body'].astype(str).values)
    test_matrix = tokenizer.transform(test_data['review_body'].astype(str).values)

    classifier = OneVsOneClassifier(LinearSVC(random_state=0))
    # LinearSVC often requires dense arrays.
    prediction = classifier.fit(train_matrix.todense(), train_data['feedback sentiment']).predict(test_matrix.todense())

    print("Confusion Matrix:")
    print(confusion_matrix(test_data['feedback sentiment'], prediction))
    print("Classification Report:")
    print(classification_report(test_data['feedback sentiment'], prediction))


# ----------------------------
# Experiment 5: Gaussian Naive Bayes
# ----------------------------
def experiment_gaussian_nb(data):
    from sklearn.naive_bayes import GaussianNB

    print("\n=== Gaussian Naive Bayes Experiment ===")
    df = data.copy()
    # Use 40k for training, 10k for testing.
    train_data = df.head(40000)
    test_data = df.tail(10000)
    train_data['feedback sentiment'] = train_data['star_rating'].apply(lambda r: 1 if r > 3.0 else 0)
    test_data['feedback sentiment'] = test_data['star_rating'].apply(lambda r: 1 if r > 3.0 else 0)

    print("Train sample:")
    print(train_data.head())
    print("Test sample:")
    print(test_data.head())

    tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = tokenizer.fit_transform(train_data['review_body'].astype(str).values)
    test_matrix = tokenizer.transform(test_data['review_body'].astype(str).values)

    nb = GaussianNB()
    nb.fit(train_matrix.todense(), train_data['feedback sentiment'])
    prediction = nb.predict(test_matrix.todense())

    print("Confusion Matrix:")
    print(confusion_matrix(test_data['feedback sentiment'], prediction))
    print("Classification Report:")
    print(classification_report(test_data['feedback sentiment'], prediction))


# ----------------------------
# Experiment 6: MLP Classifier
# ----------------------------
def experiment_mlp(data):
    from sklearn.neural_network import MLPClassifier

    print("\n=== MLP Classifier Experiment ===")
    df = data.copy()
    # Use 40k for training, 10k for testing; remove star_rating == 3.
    train_data = df.head(40000)
    test_data = df.tail(10000)
    train_data = train_data[train_data['star_rating'] != 3]
    test_data = test_data[test_data['star_rating'] != 3]

    train_data['feedback sentiment'] = train_data['star_rating'].apply(lambda r: 1 if r > 3.0 else 0)
    test_data['feedback sentiment'] = test_data['star_rating'].apply(lambda r: 1 if r > 3.0 else 0)

    print("Train sample:")
    print(train_data.head())
    print("Test sample:")
    print(test_data.head())

    tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = tokenizer.fit_transform(train_data['review_body'].astype(str).values)
    test_matrix = tokenizer.transform(test_data['review_body'].astype(str).values)

    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train_matrix.todense(), train_data['feedback sentiment'])
    prediction = clf.predict(test_matrix.todense())

    print("Confusion Matrix:")
    print(confusion_matrix(test_data['feedback sentiment'], prediction))
    print("Classification Report:")
    print(classification_report(test_data['feedback sentiment'], prediction))


# ----------------------------
# Experiment 7: Quadratic Discriminant Analysis
# ----------------------------
def experiment_quadratic_discriminant(data):
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

    print("\n=== Quadratic Discriminant Analysis Experiment ===")
    df = data.copy()
    # Use 40k for training, 10k for testing.
    train_data = df.head(40000)
    test_data = df.tail(10000)
    train_data['feedback sentiment'] = train_data['star_rating'].apply(
        lambda r: 1 if r > 3.0 else (-1 if r < 3.0 else 0))
    test_data['feedback sentiment'] = test_data['star_rating'].apply(
        lambda r: 1 if r > 3.0 else (-1 if r < 3.0 else 0))

    print("Train sample:")
    print(train_data.head())
    print("Test sample:")
    print(test_data.head())

    tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = tokenizer.fit_transform(train_data['review_body'].astype(str).values)
    test_matrix = tokenizer.transform(test_data['review_body'].astype(str).values)

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(train_matrix.todense(), train_data['feedback sentiment'])
    prediction = qda.predict(test_matrix.todense())

    print("Confusion Matrix:")
    print(confusion_matrix(test_data['feedback sentiment'], prediction))
    print("Classification Report:")
    print(classification_report(test_data['feedback sentiment'], prediction))


# ----------------------------
# Experiment 8: Linear SVC
# ----------------------------
def experiment_linear_svc(data):
    from sklearn import svm

    print("\n=== Linear SVC Experiment ===")
    df = data.copy()
    # Use 80k for training, 20k for testing.
    train_data = df.head(80000)
    test_data = df.tail(20000)
    train_data['feedback sentiment'] = train_data['star_rating'].apply(
        lambda r: 1 if r > 3.0 else (-1 if r < 3.0 else 0))
    test_data['feedback sentiment'] = test_data['star_rating'].apply(
        lambda r: 1 if r > 3.0 else (-1 if r < 3.0 else 0))

    print("Train sample:")
    print(train_data.head())
    print("Test sample:")
    print(test_data.head())

    tokenizer = CountVectorizer(token_pattern=r'\b\w+\b')
    train_matrix = tokenizer.fit_transform(train_data['review_body'].astype(str).values)
    test_matrix = tokenizer.transform(test_data['review_body'].astype(str).values)

    classifier = svm.LinearSVC()
    classifier.fit(train_matrix, train_data['feedback sentiment'])
    prediction = classifier.predict(test_matrix)

    print("Confusion Matrix:")
    print(confusion_matrix(test_data['feedback sentiment'], prediction))
    print("Classification Report:")
    print(classification_report(test_data['feedback sentiment'], prediction))


# ----------------------------
# Main: Load data and run all experiments
# ----------------------------
def main():
    # Read the CSV file once.
    data = pd.read_csv("amazon_dataframe.csv")
    
    experiment_decision_tree(data)
    experiment_knn(data)
    experiment_logistic_regression(data)
    experiment_one_vs_one_svm(data)
    experiment_gaussian_nb(data)
    experiment_mlp(data)
    experiment_quadratic_discriminant(data)
    experiment_linear_svc(data)

if __name__ == "__main__":
    main()
