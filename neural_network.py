"""
v2: Removed cross validation script for brevity's sake and to increase train speed.
    Added feature engineering.
    Removed pre-trained and custom embedding neural networks.

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

import matplotlib.pyplot as plt
import keras
import numpy as np
from numpy.random import random
import pandas as pd
from keras import layers as kn
from nltk.corpus import reuters  # https://www.nltk.org/book/ch02.html
from nltk.stem import WordNetLemmatizer
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
stop_words = set(stopwords.words('english'))


# ===========================================
# Functions
# Beware some of them might use global variables which are defined later
# ===========================================
def get_data_splits(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    return X_train, X_test, y_train, y_test


def print_confusion_matrix_sample(y_true, y_pred, label_range=range(0, 3)):
    if type(label_range) == int:
        label_range = [label_range]
    for i in label_range:
        print('Label "' + y_label[i] + '":')
        yt = np.array(y_true[:, i].T.todense())[0]
        pt = y_pred[:, i] > 0
        print("P \ A   False  True")  # Prediction vs Actual label
        print("False   {:<7n}{:n}".format(sum(~yt & ~pt), sum(yt & ~pt)))
        print("True    {:<7n}{:n}".format(sum(~yt & pt), sum(yt & pt)))


def increase_rare_feature_count(X_train, y_train, minlim=5, increment=5):
    # checking count of each label, there are labels with just 1 or 2 entries in total, to be able to stratify
    # train/test/validation we can duplicate some of these entries, as a quick and dirty fix, a "pseudo-bagging" approach.
    # Also when duplicating values, we can shuffle some of the word values to not overfit.
    print("Before adding features:", np.sum(y_train, axis=0))
    rare_label_cols = [n for n, i in enumerate(np.sum(y_train, axis=0).tolist()[0]) if i < minlim]
    rare_label_rows = [n for n, i in enumerate(np.sum(y_train[:, rare_label_cols], axis=1).T.tolist()[0]) if i > 0]

    # x5 each of these rows
    y_to_append = sparse.vstack([y_train[rare_label_rows, :]]*(increment-1))
    X_to_append = sparse.vstack([X_train[rare_label_rows, :]]*(increment-1))

    # remove 10% of all actual words
    X_to_append.data = np.array([i if int(random()*10) != 0 else 0 for i in X_to_append.data])
    X_to_append.eliminate_zeros()

    # add 0 - 10 random other words
    X_to_append = sparse.lil_matrix(X_to_append)
    for i in range(0, X_to_append.shape[0]):
        add_len = int(random() * 11)
        X_to_append[i, np.random.choice(X_to_append.shape[1], size=add_len)] = [1]*add_len

    # append
    X_adjusted = sparse.vstack([X_train, X_to_append])
    y_adjusted = sparse.vstack([y_train, y_to_append])

    # just using 20% of train data to tune hyperparameters
    print("After adding features:", np.sum(y_adjusted, axis=0))
    # some other labels also increased due to multilabels
    return X_adjusted, y_adjusted


def plot_learner(n, history):
    plt.figure(n)
    plt.title(str(n))
    plt.ylim((0, 0.1))
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    return None


def create_bag_ofwords(df):
    # Initial testing was not an improvement over linear model, so let's try better pre-processing
    # remove symbols
    df["reduced_text_nn"] = df["text"].apply(
        lambda x: re.sub(r'[!@#$%^&()_\-=/*+<>,?;:"[\]{}`~\\n]', " ", x.lower()))

    # stopwords
    df["reduced_text_nn"] = df["reduced_text_nn"].apply(
        lambda x: " ".join([w for w in word_tokenize(x) if w not in stop_words]))

    # special case for , . ' so numbers and abbreviations don't get split
    df["reduced_text_nn"] = df["reduced_text_nn"].apply(
        lambda x: re.sub(r"['\.,]", "", x))

    # spaces
    df["reduced_text_nn"] = df["reduced_text_nn"].apply(
        lambda x: re.sub(r"\s+", " ", x))

    # replace all numbers with
    df["reduced_text_nn"] = df["reduced_text_nn"].apply(
        lambda x: re.sub(r"\d+", "digits", x))

    # lemmatize
    lemmatizer = WordNetLemmatizer()
    df['reduced_text_nn'] = df['reduced_text_nn'].apply(
        lambda x: ' '.join([lemmatizer.lemmatize(w) for w in word_tokenize(x)]))

    # # stem
    # from nltk.stem.porter import PorterStemmer
    # stemmer = PorterStemmer()
    # df['reduced_text_nn'] = df['reduced_text_nn'].apply(
    #     lambda x: ' '.join([stemmer.stem(w) for w in word_tokenize(x)]))

    return df


def nn_dense(hyperparams, size_in, size_out):
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

    nn.add(kn.Dense(hyperparams[0], input_dim=size_in, use_bias=False, activation="relu"))
    nn.add(kn.Dropout(hyperparams[1]))  # for accuracy

    for layer in range(1, layers):
        nn.add(kn.Dense(hyperparams[2*layer], activation="relu"))
        nn.add(kn.Dropout(hyperparams[2*layer+1]))

    nn.add(kn.Dense(size_out, activation="sigmoid"))
    nn.compile(optimizer='adam', loss='binary_crossentropy',
               metrics=["accuracy", keras.metrics.FalseNegatives(), keras.metrics.FalsePositives()])
    # small dataset so can use high learning rate and larger batches, unless it will cause memory to overflow.
    return nn


def optimize_nn(X_tr, X_valid, y_tr, y_valid):
    """
    Try different layer combinations up to 3 dense layers with dropout and return each run's validation f1 score
    """
    global nn_ar, score_nn_bow  # so these don't get lost if function fails
    max_neuron = 1500
    max_layers = 5
    score_nn_bow = []
    nn_ar = []
    acc=0
    n=0
    while acc < 0.88:
        layers = int(random() * max_layers)
        hyperparams = (int(random() * max_neuron), random() / 2)
        for lr in range(layers):
            hyperparams += (int(random() * max_neuron), random() / 2)

        nn_ar.append(hyperparams)

        print("\nHyperparams", nn_ar[n])
        model = nn_dense(nn_ar[n], X_tr.shape[1], y_tr.shape[1])
        es = keras.callbacks.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1, patience=3)
        model.fit(X_tr, y_tr, #validation_data=(X_valid, y_valid),
                  batch_size=500, epochs=200, callbacks=[es], verbose=False)

        y_pred = model.predict(X_valid)
        score_ = f1_score(y_valid, y_pred > 0.5, average="micro")  # use validation data for choosing best params
        score_nn_bow.append(np.average(score_))
        print("Average f1 score", score_nn_bow[n])
        acc = score_nn_bow[n]
        n+=1
        # print_confusion_matrix_sample(y_test, Xl_pred, 0)
        # print_confusion_matrix_sample(y_valid, y_pred > 0.5, 0)
        # plot_learner(n, history)
    return score_nn_bow, nn_ar


def prep_data(text_series, min_df=5, minlim=5, increment=5, valid_size=0.4):
    # # TFIDF
    # from sklearn.feature_extraction.text import TfidfVectorizer
    # df = create_bag_ofwords(df)
    # nn_tf = TfidfVectorizer(min_df=min_df)
    # X_nn = nn_tf.fit_transform(text_series)

    # Count Vectorizer
    nn_cv = CountVectorizer(min_df=min_df)
    X_nn = nn_cv.fit_transform(text_series)  # replace with log to reduce bias in long texts
    X_nn.data = np.array([np.log(i + np.e) for i in X_nn.data])

    # add additional custom features
    X_nn = sparse.csr_matrix(
        sparse.hstack(
            [
                X_nn,
                df["text"].apply(lambda x: x.count(" "))[:, None],
                df["text"].apply(lambda x: len([w for w in word_tokenize(x) if w in stop_words]))[:, None],
                df["text"].apply(lambda x: len([w for w in word_tokenize(x) if w not in stop_words]))[:, None],
                df["text"].apply(lambda x: len(re.findall(".", x)))[:, None],
                df["text"].apply(lambda x: len(re.findall("[a-z]", x)))[:, None],
                df["text"].apply(lambda x: len(re.findall("[0-9]", x)))[:, None],
                df["text"].apply(lambda x: len(re.findall(r"\?", x)))[:, None],
                df["text"].apply(
                    lambda x: len(re.findall("""[!@#$%^&*()_\-=+/.,<>?;:"[\]{}`~]""", x))
                )[:, None],
            ]
        )
    )

    # split and init
    X_train, X_test, y_train, y_test = get_data_splits(X_nn, y)
    X_train, y_train = increase_rare_feature_count(X_train, y_train, minlim=minlim, increment=increment)  # 83.1%
    X_tr, X_valid, y_tr, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=1)
    return X_train, X_test, y_train, y_test, X_tr, X_valid, y_tr, y_valid


def load_embedding_file(file_path):
    # load the whole embedding into memory
    embeddings_index = dict()
    f = open(file_path, encoding="latin")
    for line in f:
        values = line.split()
        word = values[0]
        try:
            coefs = np.array(values[1:], dtype="float32")
            if len(coefs) == 100:
                embeddings_index[word] = coefs
        except Exception:
            pass
    f.close()
    print("Loaded %s word vectors." % len(embeddings_index))
    return embeddings_index


def make_embedding_matrix(embeddings_index, tokenized):
    # create a weight matrix for words in training X_train
    embedding_matrix = np.zeros((len(t.word_index) + 1, 100))
    for word, i in tokenized.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print("Created %s embedding matrix" % len(embedding_matrix))
    return embedding_matrix


# ===========================================
# Load data to DataFrame
# ===========================================
reviews = []
for fileid in reuters.fileids():
    entry_type, filename = fileid.split("/")
    reviews.append((filename, reuters.categories(fileid), entry_type, reuters.raw(fileid)))

df = pd.DataFrame(reviews, columns=["filename", "categories", "type", "text"])
df.head()

# Remove &..> tags
df["text"] = df["text"].apply(lambda x: re.sub("(&lt;)|>", "", x))
train_idx = df[df["type"] == "training"].index
test_idx = df[df["type"] == "test"].index

cv = CountVectorizer(tokenizer=lambda x: x, lowercase=False, binary=True)
y = cv.fit_transform(df["categories"]) >= 1
y_label = cv.get_feature_names()

# make basic pre-processed text column:
df["reduced_text"] = df["text"].apply(lambda x: re.sub(r"""[\d\n!@#$%^&*()_\-=+/,<>?;:"[\]{}`~]""", " ", x.lower()))
df["reduced_text"] = df["reduced_text"].apply(lambda x: re.sub("[\.']", "", x.lower()))

# ===========================================
# Baseline model, linear multi-label classifier, Bag of Words
# Let's try to beat the most basic linear model using keras
# ===========================================

lin_cv = CountVectorizer(min_df=10, max_df=0.9)
X_bow = lin_cv.fit_transform(df["reduced_text"])
Xl_train, Xl_test, y_train, y_test = get_data_splits(X_bow, y)

bm = MultiOutputClassifier(SGDClassifier())
bm.fit(Xl_train, np.array(y_train.todense()) * 1)

Xl_pred = bm.predict(Xl_test)
score_lin = f1_score(y_test, Xl_pred, average="micro")
print("Linear score", score_lin)  # 82.5%
print_confusion_matrix_sample(y_test, Xl_pred, 0)


# ===========================================
# NN model, Bag of Words
# ===========================================
X_train, X_test, y_train, y_test = get_data_splits(X_bow, y)
model = nn_dense((1000, 0.5, 800, 0.5), X_train.shape[1], y_train.shape[1])
model.fit(X_train, y_train, batch_size=500, epochs=64, verbose=True)
y_basicnn_pred = model.predict(X_test)
bestscore_basic = f1_score(y_test, y_basicnn_pred > 0.5, average="micro")
print("Linear score", score_lin)  # 82.5%
print("NN basic score", bestscore_basic)  # 87.1%


# ===========================================
# NN model, Bag of Words, hyperparameter tuning
# ===========================================
df = create_bag_ofwords(df)

# Optimize
_, _, _, _, X_tr, X_valid, y_tr, y_valid = prep_data(df["reduced_text_nn"], valid_size=0.15)
score_nn_bow, nn_ar = optimize_nn(X_tr, X_valid, y_tr, y_valid)

# Run best model
print("Using best model for test set prediction")
best = (1031, 0.26008782458319696, 795, 0.06356556954588621, 1412, 0.24505221956319218)  # @epoch 64

_, X_test, _, y_test, X_tr, _, y_tr, _ = prep_data(df["reduced_text_nn"], valid_size=0.001)
model = nn_dense(best, X_tr.shape[1], y_tr.shape[1])
model.summary()
model.fit(X_tr, y_tr, batch_size=500, epochs=64, verbose=True)  # class_weightdoes not increase final accuracy
y_pred = model.predict(X_test)
bestscore_ = f1_score(y_test, y_pred > 0.5, average="micro")

print("Linear score", score_lin)  # 82.5%
print("NN basic score", bestscore_basic)  # 87%
print("NN heavy pre-processing score", bestscore_)  # best 85.2% - means input data might have been over-pre-processed

# Check confusion matrix:
# print("\nLinear pred:")
# print_confusion_matrix_sample(y_test, Xl_pred, 0)
# print("\nNeural Net pred:")
# print_confusion_matrix_sample(y_test, y_pred > 0.5, 0)


# ===========================================
# NN model, tokenizer / pre-trained embedding
# English CoNLL17 corpus
# http://vectors.nlpl.eu/repository/
# ===========================================
embeddings_index = load_embedding_file(r"nltk_reuters_labeling\model.txt")
# It has very various types of words, just no capital letters and most symbols:
# embeddings_index["usa."]  # ok
# embeddings_index["long-term"]  # ok
# embeddings_index["knight-ridder"]  # missing
# embeddings_index["however,"]  # missing

df["text_embed"]=df["text"].apply(lambda x: re.sub(r"""[\n\d!@#$%^&*()_\-=+/<>?;:'"[\]{}`~]""", " ", x.lower()))
df["text_embed"]=df["text_embed"].apply(lambda x: re.sub(r"[,.]", "", x.lower()))
df["text_embed"] = df["text_embed"].apply(lambda x: re.sub(r"\s+", " ", x.lower()))

# Tokenize text and integer encode all documents
t = Tokenizer()
t.fit_on_texts(df.loc[train_idx, "text_embed"])  # fit on train

vocab_size = len(t.word_index) + 1
encoded_docs = t.texts_to_sequences(df["text_embed"])
lens = [len(i) for i in encoded_docs]
mdn = int(np.average(lens)) + 20
print("Avg word count per document+20:", mdn)

padded_seq = pad_sequences(encoded_docs, maxlen=mdn, padding="post")
embedding_matrix = make_embedding_matrix(embeddings_index, t)

# get_data_splits(X, y)

# # 73.3% f1 score:
# model = keras.Sequential()
# model.add(kn.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=mdn, trainable=False))
# model.add(kn.Dropout(0.2))
# model.add(kn.Conv1D(100, 10, padding='valid', activation='relu'))
# model.add(kn.Dropout(0.2))
# model.add(kn.Conv1D(100, 10, padding='valid', activation='relu'))
# model.add(kn.Flatten())
# model.add(kn.Dropout(0.2))
# model.add(kn.Dense(90, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy',
#               metrics=["accuracy", keras.metrics.FalseNegatives(), keras.metrics.FalsePositives()])
# model.summary()

# # 72.4% f1 score:
# model = keras.Sequential()
# model.add(kn.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=mdn, trainable=False))
# model.add(kn.Conv1D(100, 10, padding='same', activation='relu'))
# model.add(kn.Flatten())
# model.add(kn.Dense(500, activation='relu'))
# model.add(kn.Dropout(0.2))
# model.add(kn.Dense(90, activation='sigmoid'))
# model.compile(optimizer='adam', loss='binary_crossentropy',
#               metrics=["accuracy", keras.metrics.FalseNegatives(), keras.metrics.FalsePositives()])
# model.summary()

# 80% f1 score
inp = kn.Input(shape=(mdn,), dtype='int32')
x = kn.Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=mdn, trainable=False)(inp)
x = kn.Conv1D(100, 5, activation='relu')(x)
x = kn.MaxPooling1D(5)(x)
x = kn.Dropout(0.2)(x)
x = kn.Conv1D(100, 5, activation='relu')(x)
x = kn.MaxPooling1D(22)(x)
x = kn.Dropout(0.2)(x)
x = kn.Flatten()(x)
x = kn.Dense(128, activation='relu')(x)
x = kn.Dropout(0.05)(x)
preds = kn.Dense(90, activation='sigmoid')(x)

model = keras.Model(inp, preds)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy', keras.metrics.FalseNegatives(), keras.metrics.FalsePositives()])
model.summary()


X_train, X_test, y_train, y_test = get_data_splits(padded_seq, y)
X_tr, X_valid, y_tr, y_valid = train_test_split(X_train, y_train, train_size=0.85)

history = model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5, batch_size=100,
          class_weight=[1/i+0.5 for i in np.sum(y_train, axis=0).tolist()[0]])
y_pred = model.predict(X_test)
bestscore_embed = f1_score(y_test, y_pred > 0.5, average="micro")

plot_learner("Embed", history)

print("Linear score", score_lin)  # 82.5%
print("NN basic score", bestscore_basic)  # 87.1%
print("NN heavy pre-processing score", bestscore_)  # best 85.2%
print("NN Embed score", bestscore_embed)  # 80%
