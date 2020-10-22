from tensorflow.keras import layers
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pandas as pd
import re
import string
import numpy as np

print("Starting")

data = pd.read_csv("data/labeled_tweets.csv", index_col=0)

print(data.head(10))
print(data[data.hate_speech == 1]['tweet'])
data = data.sample(frac=1)
data = data[:1000]
tweets = data['tweet']
labels = data['class']

X_train, X_test, y_train, y_test = train_test_split(tweets, labels, test_size=0.33, random_state=42)
raw_train_ds = tf.data.Dataset.from_tensor_slices((X_train.values, y_train.values))
raw_test_ds = tf.data.Dataset.from_tensor_slices((X_test.values, y_test.values))

# for features_tensor, target_tensor in raw_train_ds:
    # print(f'features:{features_tensor} target:{target_tensor}')

def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<[^>]+>", " ")
    stripped_punc = tf.strings.regex_replace(stripped_html, f"[{re.escape(string.punctuation)}]" , "")
    stripped = tf.strings.regex_replace(stripped_punc, f"@[\w*]\s" , "")

    return stripped

print(custom_standardization("@hiter: What! @hello <br> dk"))


max_features = 20000
embedding_dim = 128
sequence_length = 500


vectorizer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode="int",
    output_sequence_length=sequence_length,
)
text_ds = raw_train_ds.map(lambda x, y: x)
vectorizer.adapt(text_ds)

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorizer(text), label


# Vectorize the data.
train_ds = raw_train_ds.map(vectorize_text)
# val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

train_ds = train_ds.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))
test_ds = test_ds.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))
# test_ds = test_ds.map(lambda x_text, x_label: (x_text, tf.expand_dims(x_label, -1)))

# Do async prefetching / buffering of the data for best performance on GPU.
train_ds = train_ds.cache().prefetch(buffer_size=10)
# val_ds = val_ds.cache().prefetch(buffer_size=10)
test_ds = test_ds.cache().prefetch(buffer_size=10)

inputs = tf.keras.Input(shape=(None,), dtype="int64")

# print(inputs.shape)


# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(max_features, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=1)(x)
# x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a single unit output layer, and squash it with a sigmoid:
predictions = layers.Dense(1, activation="sigmoid", name="predictions")(x)

model = tf.keras.Model(inputs, predictions)

# Compile the model with binary crossentropy loss and an adam optimizer.
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

epochs = 1

# Fit the model using the train and test datasets.
model.fit(train_ds, validation_data=test_ds, epochs=epochs)
result = model.evaluate(test_ds)


# A string input
inputs = tf.keras.Input(shape=(1,), dtype="string")
# Turn strings into vocab indices
indices = vectorizer(inputs)
# Turn vocab indices into predictions
outputs = model(indices)

# Our end to end model
end_to_end_model = tf.keras.Model(inputs, outputs)
end_to_end_model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"]
)

# Test it with `raw_test_ds`, which yields raw strings
# end_to_end_model.evaluate(test_ds)

print(end_to_end_model.predict(np.array(["hello", "bitch"])))
