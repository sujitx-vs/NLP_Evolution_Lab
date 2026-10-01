import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import Input, Embedding, LSTM, Dense #type: ignore
from tensorflow.keras.models import Model #type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer #type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type: ignore

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = "datasets/manglish_english.csv"

MODEL_DIR = Path("models/manglish_seq2seq")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDING_DIM = 128
LATENT_DIM = 256

BATCH_SIZE = 16
EPOCHS = 50

RANDOM_STATE = 42


# ==========================================================
# TEXT NORMALIZATION
# ==========================================================

def normalize_text(text):
    """
    Basic normalization for this learning experiment.
    """

    text = str(text).lower().strip()

    # Keep letters, numbers, apostrophes and spaces.
    text = re.sub(r"[^a-z0-9' ]+", " ", text)

    # Remove repeated spaces.
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(DATASET_PATH)

df = df.dropna(
    subset=["manglish", "english"]
).drop_duplicates()

df["manglish"] = df["manglish"].apply(normalize_text)
df["english"] = df["english"].apply(normalize_text)

print("\nDataset size:", len(df))

print("\nSample:")
print(df.head())


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train_raw, y_test_raw = train_test_split(
    df["manglish"],
    df["english"],
    test_size=0.20,
    random_state=RANDOM_STATE
)


# ==========================================================
# ADD START / END TOKENS TO TARGET
# ==========================================================

y_train = y_train_raw.apply(
    lambda sentence: f"sos {sentence} eos"
)

y_test = y_test_raw.apply(
    lambda sentence: f"sos {sentence} eos"
)


# ==========================================================
# SOURCE TOKENIZER
# Manglish vocabulary
# ==========================================================

source_tokenizer = Tokenizer(
    oov_token="oov",
    filters=""
)

source_tokenizer.fit_on_texts(X_train)


# ==========================================================
# TARGET TOKENIZER
# English vocabulary
# ==========================================================

target_tokenizer = Tokenizer(
    oov_token="oov",
    filters=""
)

target_tokenizer.fit_on_texts(y_train)


source_vocab_size = len(
    source_tokenizer.word_index
) + 1

target_vocab_size = len(
    target_tokenizer.word_index
) + 1


print("\nManglish vocabulary size:",
      source_vocab_size)

print("English vocabulary size:",
      target_vocab_size)


# ==========================================================
# CONVERT SOURCE TEXT TO TOKEN IDS
# ==========================================================

encoder_train_sequences = (
    source_tokenizer.texts_to_sequences(X_train)
)

encoder_test_sequences = (
    source_tokenizer.texts_to_sequences(X_test)
)


# ==========================================================
# CONVERT TARGET TEXT TO TOKEN IDS
# ==========================================================

target_train_sequences = (
    target_tokenizer.texts_to_sequences(y_train)
)

target_test_sequences = (
    target_tokenizer.texts_to_sequences(y_test)
)


# ==========================================================
# FIND MAXIMUM SEQUENCE LENGTHS
# ==========================================================

all_source_sequences = (
    encoder_train_sequences +
    encoder_test_sequences
)

all_target_sequences = (
    target_train_sequences +
    target_test_sequences
)

max_source_length = max(
    len(sequence)
    for sequence in all_source_sequences
)

max_target_length = max(
    len(sequence)
    for sequence in all_target_sequences
)

# Decoder input/target is shifted by one token.
max_decoder_length = max_target_length - 1


print("\nMaximum Manglish length:",
      max_source_length)

print("Maximum English length:",
      max_target_length)


# ==========================================================
# PAD ENCODER INPUTS
# ==========================================================

encoder_train_data = pad_sequences(
    encoder_train_sequences,
    maxlen=max_source_length,
    padding="post",
    truncating="post"
)

encoder_test_data = pad_sequences(
    encoder_test_sequences,
    maxlen=max_source_length,
    padding="post",
    truncating="post"
)


# ==========================================================
# PREPARE DECODER INPUT AND TARGET
# ==========================================================
#
# Example:
#
# Full target:
#
# sos how are you bro eos
#
#
# Decoder input:
#
# sos how are you bro
#
#
# Decoder target:
#
# how are you bro eos
#
# ==========================================================


def prepare_decoder_data(sequences):

    decoder_inputs = []
    decoder_targets = []

    for sequence in sequences:

        decoder_inputs.append(
            sequence[:-1]
        )

        decoder_targets.append(
            sequence[1:]
        )

    decoder_inputs = pad_sequences(
        decoder_inputs,
        maxlen=max_decoder_length,
        padding="post",
        truncating="post"
    )

    decoder_targets = pad_sequences(
        decoder_targets,
        maxlen=max_decoder_length,
        padding="post",
        truncating="post"
    )

    return decoder_inputs, decoder_targets


decoder_train_data, decoder_target_train = (
    prepare_decoder_data(
        target_train_sequences
    )
)

decoder_test_data, decoder_target_test = (
    prepare_decoder_data(
        target_test_sequences
    )
)


# ==========================================================
# SAMPLE WEIGHTS
# Ignore padding during loss / accuracy
# ==========================================================

train_sample_weights = (
    decoder_target_train != 0
).astype("float32")

test_sample_weights = (
    decoder_target_test != 0
).astype("float32")


# ==========================================================
# ENCODER
# ==========================================================

encoder_inputs = Input(
    shape=(max_source_length,),
    name="encoder_inputs"
)


encoder_embedding_layer = Embedding(
    input_dim=source_vocab_size,
    output_dim=EMBEDDING_DIM,
    mask_zero=True,
    name="encoder_embedding"
)


encoder_embedding = encoder_embedding_layer(
    encoder_inputs
)


encoder_lstm = LSTM(
    LATENT_DIM,
    return_state=True,
    name="encoder_lstm"
)


encoder_output, state_h, state_c = (
    encoder_lstm(
        encoder_embedding
    )
)


encoder_states = [
    state_h,
    state_c
]


# ==========================================================
# DECODER
# ==========================================================

decoder_inputs = Input(
    shape=(max_decoder_length,),
    name="decoder_inputs"
)


decoder_embedding_layer = Embedding(
    input_dim=target_vocab_size,
    output_dim=EMBEDDING_DIM,
    mask_zero=True,
    name="decoder_embedding"
)


decoder_embedding = decoder_embedding_layer(
    decoder_inputs
)


decoder_lstm = LSTM(
    LATENT_DIM,
    return_sequences=True,
    return_state=True,
    name="decoder_lstm"
)


decoder_outputs, _, _ = decoder_lstm(
    decoder_embedding,
    initial_state=encoder_states
)


# ==========================================================
# OUTPUT LAYER
# ==========================================================

decoder_dense = Dense(
    target_vocab_size,
    activation="softmax",
    name="output_softmax"
)


decoder_outputs = decoder_dense(
    decoder_outputs
)


# ==========================================================
# TRAINING MODEL
# ==========================================================

model = Model(
    inputs=[
        encoder_inputs,
        decoder_inputs
    ],
    outputs=decoder_outputs,
    name="manglish_to_english_seq2seq"
)


# ==========================================================
# COMPILE
# ==========================================================

model.compile(
    optimizer="adam",

    loss="sparse_categorical_crossentropy",

    weighted_metrics=[
        tf.keras.metrics.SparseCategoricalAccuracy(
            name="token_accuracy"
        )
    ]
)


model.summary()


# ==========================================================
# TRAIN
# ==========================================================

history = model.fit(

    [
        encoder_train_data,
        decoder_train_data
    ],

    decoder_target_train,

    sample_weight=train_sample_weights,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_split=0.20,

    verbose=1
)


# ==========================================================
# TEST EVALUATION
# ==========================================================

results = model.evaluate(

    [
        encoder_test_data,
        decoder_test_data
    ],

    decoder_target_test,

    sample_weight=test_sample_weights,

    verbose=0
)


print("\n==============================")
print("TEST RESULTS")
print("==============================")

print(
    "Test Loss:",
    results[0]
)

print(
    "Token Accuracy:",
    results[1]
)


# ==========================================================
# CREATE ENCODER INFERENCE MODEL
# ==========================================================

encoder_model = Model(
    encoder_inputs,
    [
        state_h,
        state_c
    ],
    name="encoder_inference"
)


# ==========================================================
# CREATE DECODER INFERENCE MODEL
# ==========================================================

decoder_token_input = Input(
    shape=(1,),
    name="decoder_token_input"
)


decoder_state_h_input = Input(
    shape=(LATENT_DIM,),
    name="decoder_state_h"
)


decoder_state_c_input = Input(
    shape=(LATENT_DIM,),
    name="decoder_state_c"
)


decoder_embedding_inference = (
    decoder_embedding_layer(
        decoder_token_input
    )
)


decoder_output_inference, \
decoder_state_h_output, \
decoder_state_c_output = decoder_lstm(

    decoder_embedding_inference,

    initial_state=[
        decoder_state_h_input,
        decoder_state_c_input
    ]
)


decoder_probabilities = decoder_dense(
    decoder_output_inference
)


decoder_model = Model(

    inputs=[
        decoder_token_input,
        decoder_state_h_input,
        decoder_state_c_input
    ],

    outputs=[
        decoder_probabilities,
        decoder_state_h_output,
        decoder_state_c_output
    ],

    name="decoder_inference"
)


# ==========================================================
# TRANSLATION FUNCTION
# ==========================================================

def translate(sentence):

    sentence = normalize_text(sentence)

    sequence = (
        source_tokenizer
        .texts_to_sequences([sentence])
    )

    sequence = pad_sequences(
        sequence,
        maxlen=max_source_length,
        padding="post",
        truncating="post"
    )

    # Encode Manglish sentence.
    state_h_value, state_c_value = (
        encoder_model.predict(
            sequence,
            verbose=0
        )
    )

    sos_id = target_tokenizer.word_index["sos"]
    eos_id = target_tokenizer.word_index["eos"]

    current_token = np.array(
        [[sos_id]]
    )

    translated_words = []

    for _ in range(max_decoder_length):

        probabilities, \
        state_h_value, \
        state_c_value = (
            decoder_model.predict(
                [
                    current_token,
                    state_h_value,
                    state_c_value
                ],
                verbose=0
            )
        )

        predicted_token_id = int(
            np.argmax(
                probabilities[0, 0]
            )
        )

        if predicted_token_id == eos_id:
            break

        predicted_word = (
            target_tokenizer
            .index_word
            .get(predicted_token_id)
        )

        if predicted_word is None:
            break

        if predicted_word not in {
            "sos",
            "eos",
            "oov"
        }:
            translated_words.append(
                predicted_word
            )

        # Feed prediction back to decoder.
        current_token = np.array(
            [[predicted_token_id]]
        )

    return " ".join(
        translated_words
    )


# ==========================================================
# SEQUENCE LEVEL EVALUATION
# ==========================================================

references = []
hypotheses = []

correct_sentences = 0


print("\n==============================")
print("SAMPLE TEST TRANSLATIONS")
print("==============================")


for manglish, expected in zip(
    X_test,
    y_test_raw
):

    prediction = translate(
        manglish
    )

    expected_clean = normalize_text(
        expected
    )

    prediction_clean = normalize_text(
        prediction
    )


    print("\nManglish :", manglish)
    print("Expected :", expected_clean)
    print("Predicted:", prediction_clean)


    references.append(
        [
            expected_clean.split()
        ]
    )

    hypotheses.append(
        prediction_clean.split()
    )


    if prediction_clean == expected_clean:

        correct_sentences += 1


# ==========================================================
# EXACT MATCH ACCURACY
# ==========================================================

exact_match_accuracy = (
    correct_sentences /
    len(X_test)
)


# ==========================================================
# BLEU SCORE
# ==========================================================

smoothing = (
    SmoothingFunction()
    .method1
)

bleu_score = corpus_bleu(
    references,
    hypotheses,
    smoothing_function=smoothing
)


print("\n==============================")
print("SEQUENCE METRICS")
print("==============================")

print(
    "Exact Match Accuracy:",
    exact_match_accuracy
)

print(
    "Corpus BLEU:",
    bleu_score
)


# ==========================================================
# SAVE MODELS
# ==========================================================

model.save(
    MODEL_DIR /
    "training_model.keras"
)

encoder_model.save(
    MODEL_DIR /
    "encoder_model.keras"
)

decoder_model.save(
    MODEL_DIR /
    "decoder_model.keras"
)


# ==========================================================
# SAVE TOKENIZERS + CONFIG
# ==========================================================

artifacts = {

    "source_tokenizer":
        source_tokenizer,

    "target_tokenizer":
        target_tokenizer,

    "max_source_length":
        max_source_length,

    "max_decoder_length":
        max_decoder_length,

    "embedding_dim":
        EMBEDDING_DIM,

    "latent_dim":
        LATENT_DIM
}


with open(
    MODEL_DIR / "tokenizers.pkl",
    "wb"
) as file:

    pickle.dump(
        artifacts,
        file
    )


print(
    "\nModels and tokenizers saved to:",
    MODEL_DIR
)


# ==========================================================
# REAL-TIME TRANSLATION
# ==========================================================

print("\n==============================")
print("MANGLISH → ENGLISH TRANSLATOR")
print("==============================")

print(
    "Type 'quit' to exit."
)


while True:

    user_input = input(
        "\nManglish: "
    ).strip()

    if user_input.lower() == "quit":
        break

    translation = translate(
        user_input
    )

    print(
        "English:",
        translation
    )