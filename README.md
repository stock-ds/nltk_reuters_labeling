# NLTK Reuters Labeling

Trying neural networks for NLTK Reuters news dataset's label predictions.

Full main script in `neural_netrowk.py`

Some additional code snippets and different, worse ANN approaches in `discarded.py`

## Text processing pipeline

First, data is loaded into a dataframe for easier processing. 

```python
reviews = []
for fileid in reuters.fileids():
    entry_type, filename = fileid.split("/")
    reviews.append(
        (filename, reuters.categories(fileid), entry_type, reuters.raw(fileid))
    )
df = pd.DataFrame(reviews, columns=["filename", "categories", "type", "text"])
```

Text is pre-processed by removing itallic tags `&lt;` to have only actual text.

```python
df["text"] = df["text"].apply(lambda x: re.sub("&.{1,20}>", "", x))
```

Also, all words were lowercased, symbols replaced by spaces, `.` removed.

```python
# make basic pre-processed text column:
df["reduced_text"] = df["text"].apply(lambda x: re.sub(r"""[\d\n!@#$%^&*()_\-=+/,<>?;:'"[\]{}`~]""", " ", x.lower()))
df["reduced_text"] = df["reduced_text"].apply(lambda x: re.sub("\.", "", x.lower()))
```

Text is split into train / test data sets following the
original NLTK split. Train data set is further split into 
train/validation data where necessary.

## Linear Bag Of Words Model

First, we set the most basic sklearn multi label model
based on SGDClassifier as a base line to beat using a neural network. 
It was trained in seconds and gives micro f1 score of 82.5%.

```python
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

## Dense Neural Network

Dense neural network being one of the most basic ANN models, let's see if it can perform
better than a traditional basic machine learning approach - a linear model.

Just as with the linear model, let's use the same data and some default parameters.
The model is:
`Input - Dense(1000) - Dropout(0.5) - Dense(800) - Dropout(0.5) - output Dense(90)`

There are 2 layers and dropouts to decrease overfitting. 1000 and 800 layers are just low 
numbers between input dimensions and output. The problem is likely not difficult enough
to need more layers or neurons.

```python
X_train, X_test, y_train, y_test = get_data_splits(X_bow, y)
model = nn_dense((1000, 0.5, 800, 0.5), X_train.shape[1], y_train.shape[1])
model.fit(X_train, y_train, batch_size=500, epochs=64, verbose=True)
y_basicnn_pred = model.predict(X_test)
bestscore_basic = f1_score(y_test, y_basicnn_pred > 0.5, average="micro")
print("Linear score", score_lin)  # 82.5%
print("NN basic score", bestscore_basic)  # 87.1%
```

This gives accuracy of 87.1%.

## Dense Neural Network feature engineering and optimization

Let's see if adding additional additional features and optimizing the network architecture
will increase the final model accuracy.

Text pre-processing:
* remove symbols
* special case for , . ' so numbers and abbreviations don't get split
* remove stopwords
* remove excess spaces
* replace all numbers with
* lemmatization

Features added:
* Count of spaces
* Count of capital letters
* Count of non-capital letters
* Count of digits
* Count of special symbols

```python
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
print("NN heavy pre-processing score", bestscore_)  # best 85.2%

```

This gives accuracy of 85.2%, which did not increase the basic model accuracy and even reduced it. 
This was most likely the result of over-pre-procesing. Neural networks often work best with as
little pre-processing as possible as they can figure out that internally.

## Embedding Neural Network

Pre-trained embedding: The data set might too small to create a relevant embedding layer
especially considering almost half of all the data has only `['earn']` category. So I 
also tried using English CoNLL17 corpus from http://vectors.nlpl.eu/repository/

```python
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
```

This gave accuracy of 80% which is a downgrade from a basic model. It might increase with 
a better network architecture, but it might as well not. The over complicated embedding with
deep neural network approach turns out to not be necessary for this data.

______

### Bonus: Different Artificial Neural Network approaches

Embedding: There are other neural network approaches and architectures tried in 
`nn_dense() @ discarded.py` but none
 of them worked out and all returns lower that linear model accuracy,
so it might not be worth further using this stragegy. Pretty much all models were in the 60% - 75% range.
Considering embedding approach takes much longer to train (on a non-gpu system) and the initial 
testing gave up to 70% accuracy. 
