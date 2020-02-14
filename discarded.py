
import random
import re

import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras import backend as K
import numpy as np
from numpy.random import random
import pandas as pd
from keras import layers as kn
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from nltk.corpus import reuters  # https://www.nltk.org/book/ch02.html
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import KFold
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))


def custom_loss_sigmoid_focal_crossentropy():
    def sigmoid_focal_crossentropy(y_true,
                                   y_pred,
                                   alpha=0.25,
                                   gamma=2.0,
                                   from_logits=False):
        """
        From https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        Does not penalize false negatives enough and too false-positive-averse, too long to converge
        No improvement over binary crossentropy
        """
        ce = K.binary_crossentropy(y_true, y_pred, from_logits=from_logits)

        p_t = (y_true * y_pred) + ((1 - y_true) * (1 - y_pred))
        alpha_factor = 1.0
        modulating_factor = 1.0

        if alpha:
            alpha = tf.convert_to_tensor(alpha, dtype=K.floatx())
            alpha_factor = (y_true * alpha + (1 - y_true) * (1 - alpha))

        if gamma:
            gamma = tf.convert_to_tensor(gamma, dtype=K.floatx())
            modulating_factor = tf.pow((1.0 - p_t), gamma)

        # compute the final loss and return
        return tf.reduce_sum(alpha_factor * modulating_factor * ce, axis=-1)
    return sigmoid_focal_crossentropy


def custom_loss_fbeta(beta=2):
    """
    From https://www.kaggle.com/arsenyinfo/f-beta-score-for-keras#L29
    Did not converge
    """
    #
    def fbeta(y_true, y_pred):
        y_pred_bin = K.clip(y_pred, 0, 1)
        eps = 1e-07
        tp = K.sum(y_true * y_pred_bin, axis=1) + eps
        fp = K.sum(y_pred_bin * (1 - y_true), axis=1)
        fn = K.sum((1 - y_pred_bin) * y_true, axis=1)
        # tp = K.sum(y_true * y_pred_bin) + eps
        # fp = K.sum(y_pred_bin * (1 - y_true))
        # fn = K.sum((1 - y_pred_bin) * y_true)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)

        beta_squared = beta ** 2
        return K.mean((beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall + eps))

    return fbeta

def nn_dense_tests(hyperparams, size_in, size_out):

    layers = round(len(hyperparams)/2)
    nn = keras.Sequential()

    nn.add(kn.Dense(hyperparams[0], input_dim=size_in, use_bias=False, activation="relu"))
    nn.add(kn.Dropout(hyperparams[1]))  # for accuracy
    # nn.add(kn.BatchNormalization())  # for speed - but decreased accuracy

    for layer in range(1, layers):
        nn.add(kn.Dense(hyperparams[2*layer], activation="relu"))
        nn.add(kn.Dropout(hyperparams[2*layer+1]))
        # nn.add(kn.BatchNormalization())

    # nn.add(kn.Dense(size_out, activation="sigmoid"))
    # nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005), loss=["sigmoid", "hamming"], metrics=["accuracy"])
    nn.add(kn.Dense(size_out, activation="sigmoid"))

    # nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=custom_loss_fbeta(), metrics=["accuracy"])

    # nn.compile(optimizer=keras.optimizers.Adam(learning_rate=0.01), loss=custom_loss_sigmoid_focal_crossentropy(),
    #            metrics=["accuracy", keras.metrics.FalseNegatives(), keras.metrics.FalsePositives()])

    nn.compile(optimizer='adam', loss='binary_crossentropy',
               metrics=["accuracy", keras.metrics.FalseNegatives(), keras.metrics.FalsePositives()])

    # small dataset so can use high learning rate and larger batches, unless it will cause memory to overflow.
    # print(nn.summary())
    return nn



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


def get_data_splits(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


def nn_dense(hyperparams, size_in, size_out, embed=0):
    """
    Return compiled neural network of specified size.

    Args:
        hyperparams (tuple length >2):

            1) neuron count in 1st layer
            2) 1st dropout between 0 and 1

            3) neuron count in 2nd layer
            4) 2nd dropout between 0 and 1

            5) ...

        size_in: input layer dimension
        size_out: output layer dimension

    Returns:
        neural network of specified architecture with
            sigmoid activation output,
            adam opnimizer,
            binary_crossentropy loss
    """

    layers = round(len(hyperparams)/2)
    nn = keras.Sequential()

    if embed > 0:
        # vocab_size as vocabulary size
        # embed[1] embed vector size
        nn.add(kn.Embedding(vocab_size, embed, input_length=size_in))
        nn.add(kn.Dense(hyperparams[0], activation="relu"))
        nn.add(kn.Dropout(hyperparams[1]))  # for accuracy
    else:
        nn.add(kn.Dense(hyperparams[0], input_dim=size_in, use_bias=False, activation="relu"))
        nn.add(kn.Dropout(hyperparams[1]))  # for accuracy

    for layer in range(1, layers):
        nn.add(kn.Dense(hyperparams[2*layer], activation="relu"))
        nn.add(kn.Dropout(hyperparams[2*layer+1]))

    if embed > 0:
        nn.add(kn.Flatten())

    nn.add(kn.Dense(size_out, activation="sigmoid"))
    nn.compile(optimizer='adam', loss='binary_crossentropy',
               metrics=["accuracy", keras.metrics.FalseNegatives(), keras.metrics.FalsePositives()])
    # small dataset so can use high learning rate and larger batches, unless it will cause memory to overflow.
    return nn

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


# ===========================================
# NN model, tokenizer / pre-trained embedding, Gigaword
# ===========================================
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
t.fit_on_texts(df.loc[train_idx, "reduced_text_nn"])  # fit on train

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(df["text"])
lens = [len(i) for i in encoded_docs]
mdn = int(np.average(lens))
print("Avg word count per document:", mdn)

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


# ===========================================
# NN, custom embedding
# ===========================================
def hyperparams_embed():
    # same as hyperparams_nn_dense() just with 1 more random variable added for embeding dimension <300
    yield (int(random()*300), int(random()*5000), random()/1.5, int(random()*3000), random()/1.5)


t = Tokenizer()
t.fit_on_texts(df.loc[train_idx, "reduced_text_nn"])  # fit on train
vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(df["text"])
lens = [len(i) for i in encoded_docs]
mdn = int(np.average(lens))
print("Avg word count per document:", mdn)
maxlen = 70
# pad documents to a max length of 70 because average document is 53
padded_seq = pad_sequences(encoded_docs, maxlen=maxlen, padding="post")

X_train, X_test, y_train, y_test = get_data_splits(padded_seq, y)
# X_train, y_train = increase_rare_feature_count(X_train, y_train)
X_tr, X_valid, y_tr, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=1)

print("No hyper parameter oprimization test set prediction")
# best = (150, 1027, 0.25, 390, 0.05)
nn_ar = (5, 1000, 0.2)
model = nn_dense(nn_ar[1:], X_train.shape[1], y_train.shape[1], embed=(nn_ar[0]))

# # 72% f1 score:
# model = keras.Sequential()
# model.add(kn.Embedding(vocab_size, 90, input_length=maxlen))
# model.add(kn.Dropout(0.2))
# model.add(kn.Conv1D(90, 70, padding='valid'))
# model.add(kn.Flatten())
# model.add(kn.Dropout(0.2))
# model.add(kn.Dense(y.shape[1], activation="sigmoid"))
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy", keras.metrics.FalseNegatives(), keras.metrics.FalsePositives()])
# model.summary()


es = keras.callbacks.callbacks.EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
history = model.fit(X_tr, y_tr, validation_data=(X_valid, y_valid), batch_size=100, epochs=20, callbacks=[es], verbose=True)
# history = model.fit(X_train, y_train, batch_size=100, epochs=2000, verbose=False)
y_pred = model.predict(X_test)
bestscore_ = f1_score(y_test, y_pred > 0.5, average="micro")

plot_learner(n, history)
print("Linear score", score_lin)  # 81.9%
print("NN score", bestscore_)  # 72.3%

# print("\nLinear pred:")
# print_confusion_matrix_sample(y_test, Xl_pred, 0)
# print("\nNeural Net pred:")
# print_confusion_matrix_sample(y_test, y_pred > 0.5, 0)
#
# print("\nLinear pred:")
# print_confusion_matrix_sample(y_test, Xl_pred, 4)
# print("\nNeural Net pred:")
# print_confusion_matrix_sample(y_test, y_pred > 0.5, 4)

