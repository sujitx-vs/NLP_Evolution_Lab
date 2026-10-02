import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import ( #type: ignore
    Input,
    Embedding,
    LSTM,
    Dense,
    Attention,
    Concatenate
)  # type: ignore

from tensorflow.keras.models import Model  # type: ignore

from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences  # type: ignore

from tensorflow.keras.callbacks import ( #type: ignore
    EarlyStopping,
    ReduceLROnPlateau
)  # type: ignore

from nltk.translate.bleu_score import (
    corpus_bleu,
    SmoothingFunction
)


# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = "datasets/manglish_english_6000.csv"

MODEL_DIR = Path(
    "models/manglish_seq2seq_attention"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


EMBEDDING_DIM = 64

LATENT_DIM = 128

BATCH_SIZE = 8

EPOCHS = 200

RANDOM_STATE = 42


# ==========================================================
# TEXT NORMALIZATION
# ==========================================================

def normalize_text(text):

    text = str(text).lower().strip()

    text = re.sub(
        r"[^a-z0-9' ]+",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ==========================================================
# LOAD DATASET
# ==========================================================

df = pd.read_csv(
    DATASET_PATH
)


df = (
    df
    .dropna(
        subset=[
            "manglish",
            "english"
        ]
    )
    .drop_duplicates()
)


df["manglish"] = (
    df["manglish"]
    .apply(normalize_text)
)


df["english"] = (
    df["english"]
    .apply(normalize_text)
)


print("\n==============================")
print("DATASET")
print("==============================")


print(
    "Dataset size:",
    len(df)
)


print(
    df.head()
)


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train_raw, y_test_raw = train_test_split(

    df["manglish"],

    df["english"],

    test_size=0.20,

    random_state=RANDOM_STATE
)


print(
    "\nTraining samples:",
    len(X_train)
)


print(
    "Testing samples:",
    len(X_test)
)


# ==========================================================
# ADD SOS / EOS
# ==========================================================

y_train = y_train_raw.apply(

    lambda sentence:
    f"sos {sentence} eos"
)


y_test = y_test_raw.apply(

    lambda sentence:
    f"sos {sentence} eos"
)


# ==========================================================
# SOURCE TOKENIZER
# ==========================================================

source_tokenizer = Tokenizer(

    oov_token="oov",

    filters=""
)


source_tokenizer.fit_on_texts(
    X_train
)


# ==========================================================
# TARGET TOKENIZER
# ==========================================================

target_tokenizer = Tokenizer(

    oov_token="oov",

    filters=""
)


target_tokenizer.fit_on_texts(
    y_train
)


source_vocab_size = (

    len(
        source_tokenizer.word_index
    )

    + 1
)


target_vocab_size = (

    len(
        target_tokenizer.word_index
    )

    + 1
)


print("\n==============================")
print("VOCABULARY")
print("==============================")


print(
    "Manglish vocabulary:",
    source_vocab_size
)


print(
    "English vocabulary:",
    target_vocab_size
)


# ==========================================================
# TEXT -> TOKEN IDS
# ==========================================================

encoder_train_sequences = (

    source_tokenizer
    .texts_to_sequences(
        X_train
    )
)


encoder_test_sequences = (

    source_tokenizer
    .texts_to_sequences(
        X_test
    )
)


target_train_sequences = (

    target_tokenizer
    .texts_to_sequences(
        y_train
    )
)


target_test_sequences = (

    target_tokenizer
    .texts_to_sequences(
        y_test
    )
)


# ==========================================================
# MAXIMUM LENGTH
# ==========================================================

all_source_sequences = (

    encoder_train_sequences
    +
    encoder_test_sequences
)


all_target_sequences = (

    target_train_sequences
    +
    target_test_sequences
)


max_source_length = max(

    len(sequence)

    for sequence
    in all_source_sequences
)


max_target_length = max(

    len(sequence)

    for sequence
    in all_target_sequences
)


max_decoder_length = (

    max_target_length - 1
)


print("\nMaximum source length:",
      max_source_length)

print("Maximum target length:",
      max_target_length)


# ==========================================================
# PAD ENCODER INPUT
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
# PREPARE DECODER INPUT / TARGET
# ==========================================================
#
# Full:
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


    return (
        decoder_inputs,
        decoder_targets
    )


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
# Ignore padding
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

    shape=(
        max_source_length,
    ),

    name="encoder_inputs"
)


# ==========================================================
# ENCODER EMBEDDING
# ==========================================================

encoder_embedding_layer = Embedding(

    input_dim=source_vocab_size,

    output_dim=EMBEDDING_DIM,

    mask_zero=True,

    name="encoder_embedding"
)


encoder_embedding = (

    encoder_embedding_layer(
        encoder_inputs
    )
)


# ==========================================================
# ENCODER LSTM
# ==========================================================
#
# BIG DIFFERENCE:
#
# return_sequences=True
#
# OLD:
#
# only final h and c
#
#
# NOW:
#
# encoder_outputs =
#
# h1, h2, h3, ... hTs
#
# Shape:
#
# (B, Ts, 128)
#
# ==========================================================

encoder_lstm = LSTM(

    LATENT_DIM,

    return_sequences=True,

    return_state=True,

    name="encoder_lstm"
)


encoder_outputs, state_h, state_c = (

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

    shape=(
        max_decoder_length,
    ),

    name="decoder_inputs"
)


decoder_embedding_layer = Embedding(

    input_dim=target_vocab_size,

    output_dim=EMBEDDING_DIM,

    mask_zero=True,

    name="decoder_embedding"
)


decoder_embedding = (

    decoder_embedding_layer(
        decoder_inputs
    )
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
# ATTENTION
# ==========================================================
#
# Query:
#
# decoder_outputs
#
# shape:
#
# (B, Tt, H)
#
#
# Value / Key:
#
# encoder_outputs
#
# shape:
#
# (B, Ts, H)
#
#
# Attention internally calculates:
#
# score[t,i] =
#
# decoder_hidden[t]
# DOT
# encoder_hidden[i]
#
#
# scores shape:
#
# (B, Tt, Ts)
#
#
# Softmax is then applied over Ts.
#
# ==========================================================

attention_layer = Attention(
    name="attention"
)


attention_context, attention_scores = (

    attention_layer(

        [
            decoder_outputs,
            encoder_outputs
        ],

        return_attention_scores=True
    )
)


# ==========================================================
# ATTENTION CONTEXT SHAPE
# ==========================================================
#
# decoder_outputs:
#
# (B, Tt, 128)
#
#
# attention_context:
#
# (B, Tt, 128)
#
#
# Combine them:
#
# (B, Tt, 256)
#
# ==========================================================

decoder_combined_context = Concatenate(
    axis=-1,
    name="decoder_attention_concat"
)(

    [
        decoder_outputs,
        attention_context
    ]
)


# ==========================================================
# OUTPUT LAYER
# ==========================================================

decoder_dense = Dense(

    target_vocab_size,

    activation="softmax",

    name="output_softmax"
)


decoder_probabilities = decoder_dense(

    decoder_combined_context
)


# ==========================================================
# TRAINING MODEL
# ==========================================================

model = Model(

    inputs=[

        encoder_inputs,

        decoder_inputs
    ],

    outputs=decoder_probabilities,

    name="manglish_seq2seq_attention"
)


# ==========================================================
# COMPILE
# ==========================================================

optimizer = tf.keras.optimizers.Adam(

    learning_rate=0.0005
)


model.compile(

    optimizer=optimizer,

    loss="sparse_categorical_crossentropy",

    weighted_metrics=[

        tf.keras.metrics
        .SparseCategoricalAccuracy(
            name="token_accuracy"
        )
    ]
)


model.summary()


# ==========================================================
# CALLBACKS
# ==========================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=25,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=5,

    min_lr=1e-6,

    verbose=1
)


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

    callbacks=[

        early_stopping,

        reduce_lr
    ],

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
# ENCODER INFERENCE MODEL
# ==========================================================
#
# IMPORTANT:
#
# Previously encoder inference returned only:
#
# h, c
#
#
# Attention also needs:
#
# ALL encoder outputs
#
# ==========================================================

encoder_model = Model(

    encoder_inputs,

    [

        encoder_outputs,

        state_h,

        state_c
    ],

    name="encoder_attention_inference"
)


# ==========================================================
# DECODER INFERENCE INPUTS
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


# ==========================================================
# ALL ENCODER STATES
# ==========================================================

encoder_outputs_input = Input(

    shape=(
        max_source_length,
        LATENT_DIM
    ),

    name="encoder_outputs_input"
)


# ==========================================================
# DECODER EMBEDDING
# ==========================================================

decoder_embedding_inference = (

    decoder_embedding_layer(
        decoder_token_input
    )
)


# ==========================================================
# ONE DECODER STEP
# ==========================================================

decoder_output_inference, \
decoder_state_h_output, \
decoder_state_c_output = decoder_lstm(

    decoder_embedding_inference,

    initial_state=[

        decoder_state_h_input,

        decoder_state_c_input
    ]
)


# ==========================================================
# ATTENTION DURING INFERENCE
# ==========================================================
#
# decoder_output_inference:
#
# (B, 1, 128)
#
#
# encoder_outputs_input:
#
# (B, Ts, 128)
#
#
# scores:
#
# (B, 1, Ts)
#
# ==========================================================

attention_context_inference, \
attention_scores_inference = attention_layer(

    [

        decoder_output_inference,

        encoder_outputs_input
    ],

    return_attention_scores=True
)


# ==========================================================
# CONCAT DECODER STATE + CONTEXT
# ==========================================================

decoder_combined_inference = Concatenate(
    axis=-1
)(

    [

        decoder_output_inference,

        attention_context_inference
    ]
)


# ==========================================================
# OUTPUT PROBABILITIES
# ==========================================================

decoder_probabilities_inference = decoder_dense(

    decoder_combined_inference
)


# ==========================================================
# DECODER INFERENCE MODEL
# ==========================================================

decoder_model = Model(

    inputs=[

        decoder_token_input,

        encoder_outputs_input,

        decoder_state_h_input,

        decoder_state_c_input
    ],

    outputs=[

        decoder_probabilities_inference,

        decoder_state_h_output,

        decoder_state_c_output,

        attention_scores_inference
    ],

    name="decoder_attention_inference"
)


# ==========================================================
# TRANSLATE
# ==========================================================

def translate(
    sentence,
    return_attention=False
):

    sentence = normalize_text(
        sentence
    )


    sequence = (

        source_tokenizer
        .texts_to_sequences(
            [sentence]
        )
    )


    sequence = pad_sequences(

        sequence,

        maxlen=max_source_length,

        padding="post",

        truncating="post"
    )


    # ======================================================
    # ENCODER
    # ======================================================

    encoder_output_values, \
    state_h_value, \
    state_c_value = encoder_model.predict(

        sequence,

        verbose=0
    )


    sos_id = (

        target_tokenizer
        .word_index["sos"]
    )


    eos_id = (

        target_tokenizer
        .word_index["eos"]
    )


    current_token = np.array(

        [
            [sos_id]
        ]
    )


    translated_words = []

    attention_history = []


    # ======================================================
    # AUTOREGRESSIVE DECODING
    # ======================================================

    for _ in range(
        max_decoder_length
    ):


        probabilities, \
        state_h_value, \
        state_c_value, \
        attention_weights = decoder_model.predict(

            [

                current_token,

                encoder_output_values,

                state_h_value,

                state_c_value
            ],

            verbose=0
        )


        predicted_token_id = int(

            np.argmax(

                probabilities[
                    0,
                    0
                ]
            )
        )


        # Save attention vector for inspection.
        attention_history.append(

            attention_weights[
                0,
                0
            ]
        )


        if (
            predicted_token_id
            == eos_id
        ):

            break


        predicted_word = (

            target_tokenizer
            .index_word
            .get(
                predicted_token_id
            )
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


        current_token = np.array(

            [
                [
                    predicted_token_id
                ]
            ]
        )


    translation = " ".join(
        translated_words
    )


    if return_attention:

        return (
            translation,
            np.array(
                attention_history
            )
        )


    return translation


# ==========================================================
# TEST TRANSLATIONS
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


    print(
        "\nManglish :",
        manglish
    )


    print(
        "Expected :",
        expected_clean
    )


    print(
        "Predicted:",
        prediction_clean
    )


    references.append(

        [

            expected_clean.split()
        ]
    )


    hypotheses.append(

        prediction_clean.split()
    )


    if (

        prediction_clean
        ==
        expected_clean

    ):

        correct_sentences += 1


# ==========================================================
# EXACT MATCH
# ==========================================================

exact_match_accuracy = (

    correct_sentences

    /

    len(X_test)
)


# ==========================================================
# BLEU
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

    MODEL_DIR
    /
    "training_model.keras"
)


encoder_model.save(

    MODEL_DIR
    /
    "encoder_model.keras"
)


decoder_model.save(

    MODEL_DIR
    /
    "decoder_model.keras"
)


# ==========================================================
# SAVE TOKENIZERS
# ==========================================================

artifacts = {

    "source_tokenizer":
        source_tokenizer,

    "target_tokenizer":
        target_tokenizer,

    "source_vocab_size":
        source_vocab_size,

    "target_vocab_size":
        target_vocab_size,

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

    MODEL_DIR
    /
    "tokenizers.pkl",

    "wb"

) as file:


    pickle.dump(

        artifacts,

        file
    )


print(
    "\nModels saved to:",
    MODEL_DIR
)


# ==========================================================
# REAL-TIME TRANSLATION
# ==========================================================

print("\n==============================")
print("MANGLISH → ENGLISH")
print("SEQ2SEQ + ATTENTION")
print("==============================")


print(
    "Type 'quit' to exit."
)


while True:


    user_input = input(
        "\nManglish: "
    ).strip()


    if (
        user_input.lower()
        == "quit"
    ):

        break


    translation, attention_matrix = translate(

        user_input,

        return_attention=True
    )


    print(
        "English:",
        translation
    )


    print(
        "Attention matrix shape:",
        attention_matrix.shape
    )