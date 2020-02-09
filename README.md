Creating a model to predict NLTK Reuters 
news dataset's labels.

Full scipt in `neural_netrowk.py`

## Text processing pipeline

First, data is loaded into a dataframe for 
easier processing. 

```
reviews = []
for fileid in reuters.fileids():
    entry_type, filename = fileid.split("/")
    reviews.append(
        (filename, reuters.categories(fileid), entry_type, reuters.raw(fileid))
    )
df = pd.DataFrame(reviews, columns=["filename", "categories", "type", "text"])
df.head()
```

Text is pre-processed by
removing itallic tags '&lt;' to have only actual text. 

```
df["text"] = df["text"].apply(lambda x: re.sub("&.{1,20}>", "", x))
```

Text is split into train / test data sets following the
original NLTK split. Train data set is also split into 
train/validation data where necessary.

## Linear Bag Of Words Model

First, we set the most basic sklearn multi label model
based on SGDClassifier as a base line to beat. It gives f1_score 81.9%

```
df["reduced_text"] = df["text"].apply(lambda x: re.sub("""[\d!@#$%^&*()_\-=+/*+.,<>,./?;:'"[\]{}`~]""", " ", x.lower()))
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
```

## Dense Neural Network Bag Of Words Model

Now let's try a basic 1-hidden-layer dense neural network model.
Most of hyperparameters will be default, but let's try tuning the number of 
neurons in hidden layer, will try `[10, 50, 300, 1000]`. Also, to increase
model's reliability when tuning hyprparameters, we will
use 3-fold cross validation for each of the tested layer size:

```
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
```

Using the best model, predict the test set to get the final accuracy. 
Most accurate model turned out to be 1000-neuron model in this case.

```
print("Using  best model for test set prediction")
model = nn_2_dense(hyperparams[best_idx])(X_train, y_train)
model.fit(X_train, y_train, batch_size=100, epochs=10, verbose=True)

y_pred = model.predict(X_test)
best_score = f1_score(y_test, y_pred > 0.5, average="micro")
print("NN score", best_score)
```

This gives accuracy of 86.1%, which is 4.2% increase over linear model.

## Dense Neural Network Bag Of Words Model with added features

Let's see if adding additional manual features will increase the final model accuracy.
Will be the best model architecture from previous section.

Features added:
Count of spaces
Count of capital letters
Count of non-capital letters
Count of digits
Count of special symbols

```
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
```

This gives accuracy of 83.4%, which did not increase the model accuracy.

## Neural Network with word vectorirez

Data set has 90 labels and 7769 training entries so it might be
difficult to increase the accuracy much more for all the labels 
due to the relatively low training data volume, especially considering half of all
the data has only `['earn']` category label which leaves 4929 
documents for all other 89 labels to train on.
``` sum(df.loc[df['type']=='training', 'categories'].apply(lambda x: x== ['earn'])) == 2840```

One way to build a better model would be to use word embeddings, using a pre-trained
word vectorizer. It also intrinsically accounts for word similarities and 
using embeddinglayer it will account for word sequences unlike bag of words.

To minimize pre-processing (no lemmatization, no lowercasing) let's use
Gigaword 5th Edition (w/o lemmatization) from http://vectors.nlpl.eu/repository/.

```
model = keras.Sequential()
model.add(kn.Embedding(vocab_size, 300, weights=[embedding_matrix], input_length=mdn, trainable=False))
***
model.add(kn.Dense(y_train.shape[1], activation="sigmoid"))
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
print(model.summary())
```

*** = Tried middle layer combinations:
Either 1) or 2):
1) `model.add(kn.Flatten())`
2) `model.add(kn.GlobalAveragePooling1D())`
* Optional: `model.add(kn.Dense(300, activation="relu"))` - did not affect model.

None of the combinations worked out and all returns lower that linear model accuracy,
so it might not be worth further using this stragegy. No model went above 71%.

Upon review, it turns out the pre-trained vectorizer might not be fit for this
data set because half of all the unique words in it were not transformed using the
vectorizer. 

`len(embedding_matrix) == 25320`
`sum([sum(i) == 0 for i in embedding_matrix]) == 10677`

Future works:
* Pre-process text to fit the pre-trained count vectorizer better.
* Train personalized word vectorizer.
* Try other architectures.
* Tune hyper parameters.
