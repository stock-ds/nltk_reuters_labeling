"""
Environment preparation:
conda create -n tf tensorflow
conda activate tf
conda install sklearn
conda install pandas
conda install keras

Python set up:
import nltk
nltk.download('reuters') # download reuters
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
"""
import random
import re

import keras
import numpy as np
import pandas as pd
from keras import layers as kn
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import reuters  # https://www.nltk.org/book/ch02.html
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier

#############################################
# Load data to DataFrame
#############################################
reviews = []
for fileid in reuters.fileids():
    entry_type, filename = fileid.split("/")
    reviews.append(
        (filename, reuters.categories(fileid), entry_type, reuters.raw(fileid))
    )
df = pd.DataFrame(reviews, columns=["filename", "categories", "type", "text"])
df.head()

# Remove &..> tags
df["text"] = df["text"].apply(lambda x: re.sub("&.{1,20}>", "", x))
train_idx = df[df["type"] == "training"].index
test_idx = df[df["type"] == "test"].index

cv = CountVectorizer(tokenizer=lambda x: x, lowercase=False, binary=True)
y = cv.fit_transform(df["categories"]) >= 1
y_label = cv.get_feature_names()


def get_data_splits(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


def print_confusion_matrix_sample(y_true, y_pred, label_range=range(0, 3)):
    if type(label_range) == int:
        label_range = [label_range]
    for i in label_range:
        print('\nLabel "' + y_label[i] + '":')
        yt = np.array(y_true[:, i].T.todense())[0]
        pt = y_pred[:, i] > 0
        print("P \ A   False  True")  # Prediction vs Actual label
        print("False   {:<7n}{:n}".format(sum(~yt & ~pt), sum(yt & ~pt)))
        print("True    {:<7n}{:n}".format(sum(~yt & pt), sum(yt & pt)))


#############################################
# Baseline model, linear multi-label classifier, Bag of Words
# Let's try to beat the most basic linear model using keras
#############################################
df["reduced_text"] = df["text"].apply(
    lambda x: re.sub("""[\d!@#$%^&*()_\-=+/*+.,<>,./?;:'"[\]{}`~]""", " ", x.lower())
)
lin_cv = CountVectorizer(min_df=10, max_df=0.9)
X_bow = lin_cv.fit_transform(df["reduced_text"])

Xl_train, Xl_test, y_train, y_test = get_data_splits(X_bow, y)
bm = MultiOutputClassifier(SGDClassifier())
yd = np.array(y_train.todense()) * 1
bm.fit(Xl_train, yd)

Xl_pred = bm.predict(Xl_test)

score_lin = f1_score(y_test, Xl_pred, average="micro")
print("Linear score", score_lin)  # 81.9%
print_confusion_matrix_sample(y_test, Xl_pred, 0)


#############################################
# NN model, Bag of Words, cross validation, hyperparameter tuning
# Using same data set as inital linear model let's see if neural networs will increase accuracy
#############################################


def nn_2_dense(hidden_size):
    def inner_(x_in, y_out):
        nn = keras.Sequential()
        nn.add(kn.Dense(hidden_size, input_dim=x_in.shape[1], activation="relu"))
        nn.add(kn.Dense(y_out.shape[1], activation="sigmoid"))
        nn.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        # nn.summary()
        return nn

    return inner_


def run_cv(model, cv_num=3):
    # create randomized indexes for random CV sampling
    train_idx_set = list(range(X_train.shape[0]))
    random.shuffle(train_idx_set)

    score_nn_bow = [0] * cv_num
    kf = KFold(n_splits=cv_num)

    for n, (train, test) in enumerate(kf.split(train_idx_set)):
        print(f"\nCV split {n+1} / 3")
        X_tr, X_valid, y_tr, y_valid = (
            X_train[train],
            X_train[test],
            y_train[train],
            y_train[test],
        )
        nn = model(X_tr, y_tr)

        es = keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min", verbose=False
        )
        nn.fit(
            X_tr,
            y_tr,
            validation_data=(X_valid, y_valid),
            batch_size=100,
            epochs=10, # short train
            callbacks=[es],
            verbose=False,
        )
        Xnn_bow_pred = nn.predict(X_valid)

        score_nn_bow[n] = f1_score(y_valid, Xnn_bow_pred > 0.5, average="micro")

    return score_nn_bow


X_train, X_test, y_train, y_test = get_data_splits(X_bow, y)

hyperparams = [10, 50, 300, 1000]
score_nn_bow = [0] * 1
for n, hp in enumerate(hyperparams):
    print("\nHyperparam", hp)
    model = nn_2_dense(hp)
    scores_ = run_cv(model, cv_num=3)

    score_nn_bow[n] = np.average(scores_)
    print("Average f1 score", score_nn_bow[n])
    # print_confusion_matrix_sample(y_valid, Xnn_bow_pred>0.5, 1)

best_idx = [n for n, i in enumerate(score_nn_bow == max(score_nn_bow)) if i == True][0]
print("\nBest f1 score", score_nn_bow[best_idx])

print("Using  best model for test set prediction")
model = nn_2_dense(hyperparams[best_idx])(X_train, y_train)
model.fit(X_train, y_train, batch_size=100, epochs=10, verbose=True)
y_pred = model.predict(X_test)
best_score = f1_score(y_test, y_pred > 0.5, average="micro")
print("Linear score", score_lin)
print("NN score", best_score)

# in - 10 - out  < linear
# in - 50 - out  < linear
# in - 300 - out  = 83.7%, better than linear 81.9%
# in - 1000 - out = 86.1%, better than linear 81.9%


#############################################
# NN model, Bag of Words, additional features
# Compare result if we added some additional features to the data:
# Count of spaces, count of capital letters, count of non-capital letters, count of digits, count of special symbols
#############################################
X_bow_fe = sparse.csr_matrix(
    sparse.hstack(
        [
            X_bow,
            df["text"].apply(lambda x: x.count(" "))[:, None],
            df["text"].apply(lambda x: len(re.findall(r"[A-Z]", x)))[:, None],
            df["text"].apply(lambda x: len(re.findall(r"[a-z]", x)))[:, None],
            df["text"].apply(lambda x: len(re.findall(r"[0-9]", x)))[:, None],
            df["text"].apply(
                lambda x: len(re.findall("""[\d!@#$%^&*()_\-=+/.,<>?;:'"[\]{}`~]""", x))
            )[:, None],
        ]
    )
)
X_train, X_test, y_train, y_test = get_data_splits(X_bow_fe, y)
model = nn_2_dense(hyperparams[best_idx])(X_train, y_train)
model.fit(X_train, y_train, batch_size=100, epochs=10, verbose=True)
y_pred = model.predict(X_test)
best_score = f1_score(y_test, y_pred > 0.5, average="micro")

print("Linear score", score_lin)
print("NN score", best_score)

# in - 1000 - out = 83.4%, didn't increase accuracy


#############################################
# NN model, tokenizer
#############################################
# Data set has 90 labels and 7769 training entries which might not be enough
# to create an accurate model from scratch. We can try using word vectorization/embedding.
# It intrinsically accounts for word similarities.
# And using embeddinglayer it will account for word sequences unlike bag of words.

# Using pre-trained library:
# Gigaword 5th Edition - newspapers dataset
# vector size: 300
# train window size: 5
# Vocabulary size: 292967
# Algorithm: Global Vectors
# Lemmatization: False
# http://vectors.nlpl.eu/repository/


def load_embedding_file(file_path):
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(file_path, encoding="utf-8")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.array(values[1:], dtype="float32")
            if len(coefs) == 300:
                embeddings_index[word] = coefs
        except Exception:
            pass
    f.close()
    print("Loaded %s word vectors." % len(embeddings_index))
    return embeddings_index


def make_embedding_matrix(embeddings_index, tokenized):
    # create a weight matrix for words in training X_train
    embedding_matrix = np.zeros((len(t.word_index) + 1, 300))
    for word, i in tokenized.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("Created %s embedding matrix" % len(embedding_matrix))
    return embedding_matrix


###############
# # Prepare documents
# # No need for the word embedding library we are using.
# df.loc[:, 'nn_processed'] = df['text'].apply(lambda x: x.lower())
# df.loc[:, 'nn_processed'] = df['nn_processed'].apply(
#     lambda x: re.sub("""[@#$%^&*()_\-=+/*+<>/;:'"[\]{}`~]""", " ", x.lower()))
# # in case using lemmatized word embeddings:
# lemmatizer = WordNetLemmatizer()
# df.loc[:, 'nn_processed'] = df['nn_processed'].apply(
#     lambda x: ' '.join([lemmatizer.lemmatize(w) for w in nltk.word_tokenize(x)]))
################

# Tokenize text and integer encode all documents
t = Tokenizer()
t.fit_on_texts(df.loc[train_idx, "text"])  # fit on train

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(df["text"])
lens = [len(i) for i in encoded_docs]
mdn = int(np.median(lens))
print("Median word count per document:", mdn)

# pad documents to a max length of 89 (median article length) words and place processed docs back into df
# it should cover the main portion of the text.
padded_seq = pad_sequences(encoded_docs, maxlen=mdn, padding="post")

X_train, X_test, y_train, y_test = get_data_splits(padded_seq, y)

embeddings_index = load_embedding_file("model.txt")
# It has very various types of words:
embeddings_index["U.S.A."]
embeddings_index["japan"]
embeddings_index["HIJACK"]
embeddings_index["Car"]
embedding_matrix = make_embedding_matrix(embeddings_index, t)


def nn_embedding():
    # Basic embedding model with Flatten
    model = keras.Sequential()
    model.add(kn.Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=mdn, trainable=False))
    model.add(kn.Flatten())
    # model.add(kn.Dense(300, activation="relu"))
    model.add(kn.Dense(y_train.shape[1], activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    return model


def nn_gap():
    # Basic embedding model with GlobalAveragePooling1D
    model = keras.Sequential()
    model.add(kn.Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=mdn, trainable=False))
    model.add(kn.GlobalAveragePooling1D())
    model.add(kn.Dense(y_train.shape[1], activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
    print(model.summary())
    return model


train_idx_set = list(range(X_train.shape[0]))
random.shuffle(train_idx_set)

X_tr, X_valid, y_tr, y_valid = train_test_split(
    X_train, y_train, test_size=0.2, random_state=1
)

for model in [nn_embedding(), nn_gap()]:
    # fit the model
    model.fit(
        X_tr, y_tr, epochs=10, verbose=0, validation_data=[X_valid, y_valid]
    )
    # model.fit(X_train, y_train, epochs=10, verbose=1)

    nn_pred = model.predict(X_test)
    nn_bool = nn_pred > 0.5

    # evaluate the model
    score_nn = f1_score(y_test, nn_bool, average="micro")

    print("Linear score", score_lin)  # 81.9%
    print("NN score", score_nn)  # nn_embedding 70%, nn_gap 63%

    print_confusion_matrix_sample(y_test, Xl_pred, 0)
    print_confusion_matrix_sample(y_test, nn_bool, 0)

print("Low model accuracy due to poor embedding.")
print("Total unique words in data set", len(embedding_matrix))  # 25320
print("Number of words not found in embedding dictionary", sum([sum(i) == 0 for i in embedding_matrix]))  # 10677

