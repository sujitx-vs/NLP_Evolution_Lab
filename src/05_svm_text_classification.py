import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 60)
print("LOAD DATASET")
print("=" * 60)

df = pd.read_csv("datasets/sentiment_dataset.csv")

print(df)


print("\n" + "=" * 60)
print("INBALANCED CHECK")
print("=" * 60)

print("duplicated rows :", df.duplicated().sum())
print("duplicated text :", df["text"].duplicated().sum())
print("label distribution :\n", df["label"].value_counts())


# remove duplicate rows
df = df.drop_duplicates(subset="text")
print("\nAfter removing duplicates\n")
print("duplicated rows :", df.duplicated().sum())

# =====================================================
# FEATURES AND LABELS
# =====================================================

print("\n" + "=" * 60)
print("FEATURES AND LABELS")
print("=" * 60)

X = df["text"]

y = df["label"]

print("\nFeatures\n")
print(X)

print("\nLabels\n")
print(y)


# =====================================================
# TRAIN TEST SPLIT
# =====================================================

print("\n" + "=" * 60)
print("TRAIN TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)


print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))


# =====================================================
# TF-IDF VECTORIZATION
# =====================================================

print("\n" + "=" * 60)
print("TF-IDF VECTORIZATION")
print("=" * 60)

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train)

X_test = vectorizer.transform(X_test)

print("Vocabulary\n")
print(vectorizer.get_feature_names_out())

print("\nTraining Shape :", X_train.shape)
print("Testing Shape  :", X_test.shape)


# =====================================================
# TRAIN SVM
# =====================================================

print("\n" + "=" * 60)
print("TRAIN SVM")
print("=" * 60)

model = LinearSVC()

model.fit(X_train, y_train)

print("Training Completed.")

# =====================================================
# PREDICTION
# =====================================================

print("\n" + "=" * 60)
print("PREDICTION")
print("=" * 60)

predictions = model.predict(X_test)

print(predictions)

# =====================================================
# ACCURACY
# =====================================================

print("\n" + "=" * 60)
print("ACCURACY")
print("=" * 60)

accuracy = accuracy_score(y_test, predictions)

print(f"Accuracy : {accuracy * 100:.2f}%")