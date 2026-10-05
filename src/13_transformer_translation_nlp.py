import re
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import ops

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import (  # type: ignore
    Input,
    Embedding,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention
)

from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

from tensorflow.keras.callbacks import ( # type: ignore
    EarlyStopping,
    ReduceLROnPlateau
)

from nltk.translate.bleu_score import (
    corpus_bleu,
    SmoothingFunction
)


# ==========================================================
# CONFIGURATION
# ==========================================================

DATASET_PATH = "datasets/manglish_english_6000.csv"

MODEL_DIR = Path(
    "models/manglish_transformer"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# TRANSFORMER CONFIGURATION
# ==========================================================

D_MODEL = 128

NUM_HEADS = 4

FF_DIM = 256

NUM_ENCODER_LAYERS = 2

NUM_DECODER_LAYERS = 2

DROPOUT_RATE = 0.1


# ==========================================================
# TRAINING CONFIGURATION
# ==========================================================

BATCH_SIZE = 32

EPOCHS = 100

RANDOM_STATE = 42

LEARNING_RATE = 0.0005


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
# ADD START / END TOKENS
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
# Manglish
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
# English
# ==========================================================

target_tokenizer = Tokenizer(

    oov_token="oov",

    filters=""
)


target_tokenizer.fit_on_texts(
    y_train
)


# ==========================================================
# VOCABULARY SIZES
# ==========================================================

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
# MAXIMUM SEQUENCE LENGTHS
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


print("\n==============================")
print("SEQUENCE LENGTHS")
print("==============================")


print(
    "Max Manglish length:",
    max_source_length
)


print(
    "Max English length:",
    max_target_length
)


print(
    "Max Decoder length:",
    max_decoder_length
)


# ==========================================================
# PAD ENCODER DATA
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
# PREPARE DECODER DATA
# ==========================================================
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
# Ignore PAD tokens during loss / accuracy
# ==========================================================

train_sample_weights = (

    decoder_target_train != 0

).astype(
    "float32"
)


test_sample_weights = (

    decoder_target_test != 0

).astype(
    "float32"
)


# ==========================================================
# POSITIONAL EMBEDDING
# ==========================================================
#
# Instead of sinusoidal positional encoding,
# this educational implementation uses
# trainable positional embeddings.
#
#
# Token:
#
#     token_embedding
#
# Position:
#
#     position_embedding
#
#
# Combined:
#
#     token_embedding + position_embedding
#
# ==========================================================

 
class PositionalEmbedding(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        vocab_size,
        max_length,
        d_model,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.vocab_size = vocab_size

        self.max_length = max_length

        self.d_model = d_model


        self.token_embedding = Embedding(

            input_dim=vocab_size,

            output_dim=d_model,

            mask_zero=True
        )


        self.position_embedding = Embedding(

            input_dim=max_length,

            output_dim=d_model
        )


    def call(
        self,
        token_ids
    ):

        sequence_length = tf.shape(
            token_ids
        )[1]


        positions = tf.range(
            start=0,
            limit=sequence_length,
            delta=1
        )


        token_vectors = (
            self.token_embedding(
                token_ids
            )
        )


        position_vectors = (
            self.position_embedding(
                positions
            )
        )


        return (

            token_vectors
            +
            position_vectors
        )


    def compute_mask(
        self,
        inputs,
        mask=None
    ):  

        return ops.not_equal(
            inputs,
            0
        )


    def get_config(self):

        config = super().get_config()

        config.update({

            "vocab_size":
                self.vocab_size,

            "max_length":
                self.max_length,

            "d_model":
                self.d_model
        })

        return config


# ==========================================================
# TRANSFORMER ENCODER BLOCK
# ==========================================================
#
# Input
#
# ->
# Multi-Head Self-Attention
#
# ->
# Residual + LayerNorm
#
# ->
# Feed Forward Network
#
# ->
# Residual + LayerNorm
#
# ==========================================================

@tf.keras.utils.register_keras_serializable()
class TransformerEncoderBlock(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout_rate=0.1,
        **kwargs
    ):

        super().__init__(**kwargs)


        self.d_model = d_model

        self.num_heads = num_heads

        self.ff_dim = ff_dim

        self.dropout_rate = dropout_rate


        # ----------------------------------------------
        # Multi-head self-attention
        # ----------------------------------------------

        self.self_attention = MultiHeadAttention(

            num_heads=num_heads,

            key_dim=(
                d_model
                //
                num_heads
            )
        )


        # ----------------------------------------------
        # Feed Forward Network
        # ----------------------------------------------

        self.ffn = tf.keras.Sequential([

            Dense(
                ff_dim,
                activation="relu"
            ),

            Dense(
                d_model
            )
        ])


        self.norm1 = LayerNormalization(
            epsilon=1e-6
        )


        self.norm2 = LayerNormalization(
            epsilon=1e-6
        )


        self.dropout1 = Dropout(
            dropout_rate
        )


        self.dropout2 = Dropout(
            dropout_rate
        )


    def call(
        self,
        inputs,
        training=False,
        mask=None
    ):

        # ==============================================
        # CREATE PADDING MASK
        # ==============================================

        attention_mask = None


        if mask is not None:

            attention_mask = (

                mask[
                    :,
                    tf.newaxis,
                    :
                ]
            )


        # ==============================================
        # SELF ATTENTION
        # ==============================================

        attention_output = (
            self.self_attention(

                query=inputs,

                value=inputs,

                key=inputs,

                attention_mask=attention_mask,

                training=training
            )
        )


        attention_output = (
            self.dropout1(

                attention_output,

                training=training
            )
        )


        # ==============================================
        # RESIDUAL + LAYER NORMALIZATION
        # ==============================================

        x = self.norm1(

            inputs
            +
            attention_output
        )


        # ==============================================
        # FEED FORWARD NETWORK
        # ==============================================

        ffn_output = self.ffn(
            x
        )


        ffn_output = (
            self.dropout2(

                ffn_output,

                training=training
            )
        )


        # ==============================================
        # RESIDUAL + LAYER NORMALIZATION
        # ==============================================

        return self.norm2(

            x
            +
            ffn_output
        )


    def compute_mask(
        self,
        inputs,
        mask=None
    ):

        return mask


    def get_config(self):

        config = super().get_config()

        config.update({

            "d_model":
                self.d_model,

            "num_heads":
                self.num_heads,

            "ff_dim":
                self.ff_dim,

            "dropout_rate":
                self.dropout_rate
        })

        return config


# ==========================================================
# TRANSFORMER DECODER BLOCK
# ==========================================================
#
# Target Input
#
# ->
# Masked Self-Attention
#
# ->
# Residual + LayerNorm
#
# ->
# Cross-Attention with Encoder
#
# ->
# Residual + LayerNorm
#
# ->
# Feed Forward
#
# ->
# Residual + LayerNorm
#
# ==========================================================

@tf.keras.utils.register_keras_serializable()
class TransformerDecoderBlock(
    tf.keras.layers.Layer
):

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout_rate=0.1,
        **kwargs
    ):

        super().__init__(**kwargs)


        self.d_model = d_model

        self.num_heads = num_heads

        self.ff_dim = ff_dim

        self.dropout_rate = dropout_rate


        # ==============================================
        # MASKED SELF ATTENTION
        # ==============================================

        self.masked_self_attention = (
            MultiHeadAttention(

                num_heads=num_heads,

                key_dim=(
                    d_model
                    //
                    num_heads
                )
            )
        )


        # ==============================================
        # CROSS ATTENTION
        # ==============================================

        self.cross_attention = (
            MultiHeadAttention(

                num_heads=num_heads,

                key_dim=(
                    d_model
                    //
                    num_heads
                )
            )
        )


        # ==============================================
        # FEED FORWARD
        # ==============================================

        self.ffn = tf.keras.Sequential([

            Dense(
                ff_dim,
                activation="relu"
            ),

            Dense(
                d_model
            )
        ])


        self.norm1 = LayerNormalization(
            epsilon=1e-6
        )


        self.norm2 = LayerNormalization(
            epsilon=1e-6
        )


        self.norm3 = LayerNormalization(
            epsilon=1e-6
        )


        self.dropout1 = Dropout(
            dropout_rate
        )


        self.dropout2 = Dropout(
            dropout_rate
        )


        self.dropout3 = Dropout(
            dropout_rate
        )


    def call(
        self,
        inputs,
        encoder_outputs,
        training=False,
        mask=None,
        encoder_mask=None
    ):

        # ==================================================
        # DECODER SELF-ATTENTION PADDING MASK
        # ==================================================

        decoder_attention_mask = None


        if mask is not None:

            decoder_attention_mask = (

                mask[
                    :,
                    tf.newaxis,
                    :
                ]
            )


        # ==================================================
        # MASKED SELF ATTENTION
        # ==================================================
        #
        # use_causal_mask=True
        #
        # prevents looking at future target tokens.
        #
        # ==================================================

        self_attention_output = (

            self.masked_self_attention(

                query=inputs,

                value=inputs,

                key=inputs,

                attention_mask=decoder_attention_mask,

                use_causal_mask=True,

                training=training
            )
        )


        self_attention_output = (
            self.dropout1(

                self_attention_output,

                training=training
            )
        )


        x = self.norm1(

            inputs
            +
            self_attention_output
        )


        # ==================================================
        # ENCODER PADDING MASK
        # ==================================================

        cross_attention_mask = None


        if encoder_mask is not None:

            cross_attention_mask = (

                encoder_mask[
                    :,
                    tf.newaxis,
                    :
                ]
            )


        # ==================================================
        # CROSS ATTENTION
        # ==================================================
        #
        # Query:
        #
        #     decoder representation
        #
        # Key:
        #
        #     encoder representations
        #
        # Value:
        #
        #     encoder representations
        #
        # ==================================================

        cross_attention_output = (

            self.cross_attention(

                query=x,

                value=encoder_outputs,

                key=encoder_outputs,

                attention_mask=cross_attention_mask,

                training=training
            )
        )


        cross_attention_output = (
            self.dropout2(

                cross_attention_output,

                training=training
            )
        )


        x2 = self.norm2(

            x
            +
            cross_attention_output
        )


        # ==================================================
        # FEED FORWARD NETWORK
        # ==================================================

        ffn_output = self.ffn(
            x2
        )


        ffn_output = (
            self.dropout3(

                ffn_output,

                training=training
            )
        )


        return self.norm3(

            x2
            +
            ffn_output
        )


    def compute_mask(
        self,
        inputs,
        mask=None
    ):

        return mask


    def get_config(self):

        config = super().get_config()

        config.update({

            "d_model":
                self.d_model,

            "num_heads":
                self.num_heads,

            "ff_dim":
                self.ff_dim,

            "dropout_rate":
                self.dropout_rate
        })

        return config


# ==========================================================
# MODEL INPUTS
# ==========================================================

encoder_inputs = Input(

    shape=(
        max_source_length,
    ),

    dtype="int32",

    name="encoder_inputs"
)


decoder_inputs = Input(

    shape=(
        max_decoder_length,
    ),

    dtype="int32",

    name="decoder_inputs"
)


# ==========================================================
# ENCODER EMBEDDING + POSITION
# ==========================================================

encoder_embedding_layer = PositionalEmbedding(

    vocab_size=source_vocab_size,

    max_length=max_source_length,

    d_model=D_MODEL,

    name="encoder_embedding"
)


encoder_x = encoder_embedding_layer(
    encoder_inputs
)


encoder_mask = encoder_embedding_layer.compute_mask(
    encoder_inputs
)


# ==========================================================
# ENCODER BLOCKS
# ==========================================================

for i in range(
    NUM_ENCODER_LAYERS
):

    encoder_x = TransformerEncoderBlock(

        d_model=D_MODEL,

        num_heads=NUM_HEADS,

        ff_dim=FF_DIM,

        dropout_rate=DROPOUT_RATE,

        name=f"encoder_block_{i + 1}"

    )(

        encoder_x,

        mask=encoder_mask
    )


encoder_outputs = encoder_x


# ==========================================================
# DECODER EMBEDDING + POSITION
# ==========================================================

decoder_embedding_layer = PositionalEmbedding(

    vocab_size=target_vocab_size,

    max_length=max_decoder_length,

    d_model=D_MODEL,

    name="decoder_embedding"
)


decoder_x = decoder_embedding_layer(
    decoder_inputs
)


decoder_mask = decoder_embedding_layer.compute_mask(
    decoder_inputs
)


# ==========================================================
# DECODER BLOCKS
# ==========================================================

for i in range(
    NUM_DECODER_LAYERS
):

    decoder_x = TransformerDecoderBlock(

        d_model=D_MODEL,

        num_heads=NUM_HEADS,

        ff_dim=FF_DIM,

        dropout_rate=DROPOUT_RATE,

        name=f"decoder_block_{i + 1}"

    )(

        decoder_x,

        encoder_outputs,

        mask=decoder_mask,

        encoder_mask=encoder_mask
    )


# ==========================================================
# VOCABULARY PROJECTION
# ==========================================================
#
# decoder_x:
#
# (B, T_target, D_MODEL)
#
#
# Dense:
#
# D_MODEL -> target_vocab_size
#
#
# Output:
#
# (B, T_target, target_vocab_size)
#
# ==========================================================

decoder_probabilities = Dense(

    target_vocab_size,

    activation="softmax",

    name="vocabulary_projection"

)(

    decoder_x
)


# ==========================================================
# FINAL MODEL
# ==========================================================

model = Model(

    inputs=[

        encoder_inputs,

        decoder_inputs
    ],

    outputs=decoder_probabilities,

    name="manglish_transformer"
)


# ==========================================================
# COMPILE
# ==========================================================

optimizer = tf.keras.optimizers.Adam(

    learning_rate=LEARNING_RATE
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

    patience=15,

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
# TRANSLATION FUNCTION
# ==========================================================
#
# Transformer inference is autoregressive.
#
# Unlike the LSTM version:
#
# We do NOT manually carry h and c.
#
# Instead:
#
# Re-run decoder with all generated tokens so far.
#
#
# Example:
#
# sos
#
# ->
# predict "how"
#
#
# sos how
#
# ->
# predict "are"
#
#
# sos how are
#
# ->
# predict "you"
#
# ==========================================================

def translate(sentence):

    sentence = normalize_text(
        sentence
    )


    # ======================================================
    # SOURCE TOKENIZATION
    # ======================================================

    source_sequence = (

        source_tokenizer
        .texts_to_sequences(
            [sentence]
        )
    )


    source_sequence = pad_sequences(

        source_sequence,

        maxlen=max_source_length,

        padding="post",

        truncating="post"
    )


    # ======================================================
    # START DECODER
    # ======================================================

    sos_id = (
        target_tokenizer
        .word_index["sos"]
    )


    eos_id = (
        target_tokenizer
        .word_index["eos"]
    )


    generated_tokens = [
        sos_id
    ]


    # ======================================================
    # AUTOREGRESSIVE GENERATION
    # ======================================================

    for _ in range(
        max_decoder_length
    ):


        decoder_sequence = pad_sequences(

            [
                generated_tokens
            ],

            maxlen=max_decoder_length,

            padding="post",

            truncating="post"
        )


        predictions = model.predict(

            [

                source_sequence,

                decoder_sequence
            ],

            verbose=0
        )


        # --------------------------------------------------
        # Current target position
        # --------------------------------------------------

        current_position = (

            len(generated_tokens)
            -
            1
        )


        predicted_token_id = int(

            np.argmax(

                predictions[
                    0,
                    current_position
                ]
            )
        )


        # --------------------------------------------------
        # EOS
        # --------------------------------------------------

        if (
            predicted_token_id
            ==
            eos_id
        ):

            break


        # --------------------------------------------------
        # Prevent endless PAD / SOS predictions
        # --------------------------------------------------

        if predicted_token_id == 0:

            break


        generated_tokens.append(
            predicted_token_id
        )


    # ======================================================
    # TOKEN IDS -> WORDS
    # ======================================================

    translated_words = []


    for token_id in generated_tokens[1:]:


        word = (

            target_tokenizer
            .index_word
            .get(
                token_id
            )
        )


        if word is None:

            continue


        if word not in {

            "sos",
            "eos",
            "oov"

        }:

            translated_words.append(
                word
            )


    return " ".join(
        translated_words
    )


# ==========================================================
# SEQUENCE EVALUATION
# ==========================================================

references = []

hypotheses = []

correct_sentences = 0


print("\n==============================")
print("TEST TRANSLATIONS")
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
# SAVE MODEL
# ==========================================================

model.save(

    MODEL_DIR
    /
    "transformer_model.keras"
)


# ==========================================================
# SAVE TOKENIZERS + CONFIG
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

    "d_model":
        D_MODEL,

    "num_heads":
        NUM_HEADS,

    "ff_dim":
        FF_DIM,

    "num_encoder_layers":
        NUM_ENCODER_LAYERS,

    "num_decoder_layers":
        NUM_DECODER_LAYERS
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
    "\nTransformer saved to:",
    MODEL_DIR
)


# ==========================================================
# REAL-TIME TRANSLATOR
# ==========================================================

print("\n==============================")
print("MANGLISH -> ENGLISH")
print("TRANSFORMER")
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
        ==
        "quit"
    ):

        break


    translation = translate(
        user_input
    )


    print(
        "English:",
        translation
    )