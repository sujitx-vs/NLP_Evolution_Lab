# ============================================================
# TINY BERT-LIKE MASKED LANGUAGE MODEL
# ============================================================
#
# Goal:
#
# Input:
#     i [MASK] playing when the rain started
#
# Predict:
#     was
#
#
# Architecture:
#
# Token IDs
#     ↓
# Token Embedding + Position Embedding
#     ↓
# Bidirectional Multi-Head Self-Attention
#     ↓
# FFN
#     ↓
# Encoder Block x N
#     ↓
# Contextual Token Representations
#     ↓
# Vocabulary Projection
#     ↓
# Predict masked tokens
#
#
# IMPORTANT:
#
# There is:
#
# NO decoder
# NO causal mask
# NO cross-attention
#
# Every token can attend to every other token.
#
# ============================================================


import re
import json
import pickle
import random
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
import keras

from keras import ops

from keras.layers import (
    Input,
    Embedding,
    Dense,
    Dropout,
    LayerNormalization,
    MultiHeadAttention
)

from keras.models import Model

from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau
)

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIG
# ============================================================

DATASET_PATH = Path(
    "datasets/sentences.csv"
)

MODEL_DIR = Path(
    "models/tiny_bert_mlm"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MODEL CONFIG
# ============================================================

MAX_LENGTH = 20

D_MODEL = 128

NUM_HEADS = 4

FF_DIM = 256

NUM_LAYERS = 2

DROPOUT_RATE = 0.1


# ============================================================
# TRAINING CONFIG
# ============================================================

BATCH_SIZE = 32

EPOCHS = 50

LEARNING_RATE = 0.0005

MASK_PROBABILITY = 0.15

RANDOM_STATE = 42


random.seed(
    RANDOM_STATE
)

np.random.seed(
    RANDOM_STATE
)

tf.random.set_seed(
    RANDOM_STATE
)


# ============================================================
# SPECIAL TOKENS
# ============================================================

PAD_TOKEN = "[PAD]"

UNK_TOKEN = "[UNK]"

MASK_TOKEN = "[MASK]"

CLS_TOKEN = "[CLS]"

SEP_TOKEN = "[SEP]"


SPECIAL_TOKENS = [

    PAD_TOKEN,
    UNK_TOKEN,
    MASK_TOKEN,
    CLS_TOKEN,
    SEP_TOKEN

]


# ============================================================
# NORMALIZE TEXT
# ============================================================

def normalize_text(text):

    text = str(text).lower().strip()

    # Keep letters, numbers and apostrophes
    words = re.findall(
        r"[a-z0-9]+(?:'[a-z0-9]+)?",
        text
    )

    return " ".join(
        words
    )


# ============================================================
# LOAD DATA
# ============================================================

if not DATASET_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found: {DATASET_PATH}"
    )


df = pd.read_csv(
    DATASET_PATH
)


if "sentence" not in df.columns:

    raise ValueError(
        "CSV must contain a column named 'sentence'"
    )


df = (
    df
    .dropna(
        subset=["sentence"]
    )
    .drop_duplicates()
)


df["sentence"] = (
    df["sentence"]
    .apply(normalize_text)
)


df = df[
    df["sentence"].str.len() > 0
]


print("\n===================================")
print("DATASET")
print("===================================")

print(
    "Sentences:",
    len(df)
)

print(
    df.head()
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

train_sentences, val_sentences = train_test_split(

    df["sentence"].tolist(),

    test_size=0.15,

    random_state=RANDOM_STATE
)


print(
    "\nTraining sentences:",
    len(train_sentences)
)

print(
    "Validation sentences:",
    len(val_sentences)
)


# ============================================================
# BUILD VOCABULARY
# ============================================================
#
# Only learn vocabulary from TRAINING data.
#
# ============================================================

word_counter = Counter()


for sentence in train_sentences:

    word_counter.update(
        sentence.split()
    )


# Sort for reproducibility
vocabulary_words = sorted(
    word_counter.keys()
)


vocabulary = (

    SPECIAL_TOKENS
    +
    vocabulary_words
)


word_to_id = {

    word: index

    for index, word
    in enumerate(vocabulary)

}


id_to_word = {

    index: word

    for word, index
    in word_to_id.items()

}


VOCAB_SIZE = len(
    vocabulary
)


PAD_ID = word_to_id[
    PAD_TOKEN
]

UNK_ID = word_to_id[
    UNK_TOKEN
]

MASK_ID = word_to_id[
    MASK_TOKEN
]

CLS_ID = word_to_id[
    CLS_TOKEN
]

SEP_ID = word_to_id[
    SEP_TOKEN
]


print("\n===================================")
print("VOCABULARY")
print("===================================")

print(
    "Vocabulary size:",
    VOCAB_SIZE
)


# ============================================================
# TOKENIZE
# ============================================================

def tokenize_sentence(sentence):

    words = normalize_text(
        sentence
    ).split()


    token_ids = [

        word_to_id.get(
            word,
            UNK_ID
        )

        for word in words

    ]


    # Add BERT-style special tokens
    token_ids = (

        [CLS_ID]
        +
        token_ids
        +
        [SEP_ID]

    )


    # Truncate
    token_ids = token_ids[
        :MAX_LENGTH
    ]


    # Ensure SEP at end if truncated
    if token_ids[-1] != SEP_ID:

        token_ids[-1] = SEP_ID


    # Pad
    padding_length = (

        MAX_LENGTH
        -
        len(token_ids)

    )


    token_ids += (

        [PAD_ID]
        *
        padding_length

    )


    return token_ids


# ============================================================
# CREATE MLM TRAINING EXAMPLE
# ============================================================
#
# Original:
#
# [CLS] i was playing outside [SEP]
#
#
# Masked:
#
# [CLS] i [MASK] playing outside [SEP]
#
#
# Labels:
#
# -100 / ignored everywhere except MASK position.
#
#
# But because Keras sample weights are easier here:
#
# y contains original token IDs
#
# sample_weight:
#
# 0 0 1 0 0 0
#
# Only masked positions contribute to loss.
#
# ============================================================

def create_masked_example(
    token_ids,
    mask_probability=0.15
):

    original_ids = np.array(
        token_ids,
        dtype=np.int32
    )


    masked_ids = original_ids.copy()


    sample_weights = np.zeros(
        len(token_ids),
        dtype=np.float32
    )


    candidate_positions = [

        i

        for i, token_id
        in enumerate(token_ids)

        if token_id not in {

            PAD_ID,
            CLS_ID,
            SEP_ID

        }

    ]


    if not candidate_positions:

        return (
            masked_ids,
            original_ids,
            sample_weights
        )


    # --------------------------------------------------------
    # Select approximately 15% of normal tokens
    # --------------------------------------------------------

    number_to_mask = max(

        1,

        int(
            round(
                len(candidate_positions)
                *
                mask_probability
            )
        )

    )


    masked_positions = random.sample(

        candidate_positions,

        min(
            number_to_mask,
            len(candidate_positions)
        )

    )


    for position in masked_positions:

        # The correct target is the original token.
        sample_weights[
            position
        ] = 1.0


        # For this learning experiment,
        # always replace selected token with [MASK].
        #
        # Original BERT used a slightly more complex
        # 80/10/10 corruption strategy.
        masked_ids[
            position
        ] = MASK_ID


    return (

        masked_ids,

        original_ids,

        sample_weights

    )


# ============================================================
# PREPARE DATASET
# ============================================================

def prepare_dataset(sentences):

    X = []

    y = []

    weights = []


    for sentence in sentences:

        token_ids = tokenize_sentence(
            sentence
        )


        masked_ids, labels, sample_weights = (
            create_masked_example(
                token_ids,
                MASK_PROBABILITY
            )
        )


        X.append(
            masked_ids
        )

        y.append(
            labels
        )

        weights.append(
            sample_weights
        )


    return (

        np.asarray(
            X,
            dtype=np.int32
        ),

        np.asarray(
            y,
            dtype=np.int32
        ),

        np.asarray(
            weights,
            dtype=np.float32
        )

    )


X_train, y_train, train_weights = prepare_dataset(
    train_sentences
)


X_val, y_val, val_weights = prepare_dataset(
    val_sentences
)


print("\n===================================")
print("TRAINING DATA")
print("===================================")

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "train_weights:",
    train_weights.shape
)


# ============================================================
# SHOW TRAINING EXAMPLE
# ============================================================

def ids_to_tokens(ids):

    return [

        id_to_word.get(
            int(token_id),
            UNK_TOKEN
        )

        for token_id in ids

    ]


print(
    "\nMasked example:"
)

print(
    ids_to_tokens(
        X_train[0]
    )
)


print(
    "\nOriginal target:"
)

print(
    ids_to_tokens(
        y_train[0]
    )
)


print(
    "\nLoss positions:"
)

print(
    train_weights[0]
)


# ============================================================
# TOKEN + POSITION EMBEDDING
# ============================================================

@keras.utils.register_keras_serializable()
class TokenAndPositionEmbedding(
    keras.layers.Layer
):

    def __init__(
        self,
        vocab_size,
        max_length,
        d_model,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


        self.vocab_size = (
            vocab_size
        )

        self.max_length = (
            max_length
        )

        self.d_model = (
            d_model
        )


        self.token_embedding = Embedding(

            input_dim=vocab_size,

            output_dim=d_model
        )


        self.position_embedding = Embedding(

            input_dim=max_length,

            output_dim=d_model
        )


    def call(
        self,
        token_ids
    ):

        sequence_length = ops.shape(
            token_ids
        )[1]


        positions = ops.arange(
            0,
            sequence_length
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


    def get_config(
        self
    ):

        config = (
            super()
            .get_config()
        )


        config.update({

            "vocab_size":
                self.vocab_size,

            "max_length":
                self.max_length,

            "d_model":
                self.d_model

        })


        return config


# ============================================================
# BERT-LIKE ENCODER BLOCK
# ============================================================
#
# IMPORTANT DIFFERENCE FROM DECODER-ONLY:
#
# There is NO:
#
# use_causal_mask=True
#
#
# Every token can attend to:
#
# left context
# +
# right context
#
# ============================================================

@keras.utils.register_keras_serializable()
class BertEncoderBlock(
    keras.layers.Layer
):

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout_rate=0.1,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


        self.d_model = d_model

        self.num_heads = num_heads

        self.ff_dim = ff_dim

        self.dropout_rate = (
            dropout_rate
        )


        # ----------------------------------------------------
        # BIDIRECTIONAL SELF ATTENTION
        # ----------------------------------------------------

        self.self_attention = (
            MultiHeadAttention(

                num_heads=num_heads,

                key_dim=(
                    d_model
                    //
                    num_heads
                )
            )
        )


        # ----------------------------------------------------
        # FEED FORWARD
        # ----------------------------------------------------

        self.ffn = keras.Sequential([

            Dense(
                ff_dim,
                activation="gelu"
            ),

            Dense(
                d_model
            )

        ])


        self.norm1 = (
            LayerNormalization(
                epsilon=1e-6
            )
        )


        self.norm2 = (
            LayerNormalization(
                epsilon=1e-6
            )
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
        training=False
    ):

        # ----------------------------------------------------
        # SELF-ATTENTION
        #
        # No causal mask!
        # ----------------------------------------------------

        attention_output = (

            self.self_attention(

                query=inputs,

                key=inputs,

                value=inputs,

                training=training

            )

        )


        attention_output = (

            self.dropout1(

                attention_output,

                training=training
            )

        )


        x = self.norm1(

            inputs
            +
            attention_output

        )


        # ----------------------------------------------------
        # FFN
        # ----------------------------------------------------

        ffn_output = self.ffn(
            x
        )


        ffn_output = (

            self.dropout2(

                ffn_output,

                training=training
            )

        )


        return self.norm2(

            x
            +
            ffn_output

        )


    def get_config(
        self
    ):

        config = (
            super()
            .get_config()
        )


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


# ============================================================
# BUILD MODEL
# ============================================================

inputs = Input(

    shape=(MAX_LENGTH,),

    dtype="int32",

    name="token_ids"
)


# ============================================================
# EMBEDDINGS
# ============================================================

x = TokenAndPositionEmbedding(

    vocab_size=VOCAB_SIZE,

    max_length=MAX_LENGTH,

    d_model=D_MODEL,

    name="token_position_embedding"

)(inputs)


x = Dropout(
    DROPOUT_RATE
)(x)


# ============================================================
# ENCODER BLOCKS
# ============================================================

for i in range(
    NUM_LAYERS
):

    x = BertEncoderBlock(

        d_model=D_MODEL,

        num_heads=NUM_HEADS,

        ff_dim=FF_DIM,

        dropout_rate=DROPOUT_RATE,

        name=f"bert_encoder_{i + 1}"

    )(x)


# ============================================================
# MLM VOCABULARY PROJECTION
# ============================================================
#
# Every token representation:
#
# D_MODEL = 128
#
# becomes:
#
# VOCAB_SIZE logits
#
#
# But loss only applies at masked positions.
#
# ============================================================

logits = Dense(

    VOCAB_SIZE,

    name="mlm_vocabulary_projection"

)(x)


model = Model(

    inputs=inputs,

    outputs=logits,

    name="tiny_bert_mlm"
)


# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=keras.optimizers.Adam(

        learning_rate=LEARNING_RATE
    ),

    loss=keras.losses
    .SparseCategoricalCrossentropy(

        from_logits=True
    ),

    weighted_metrics=[

        keras.metrics
        .SparseCategoricalAccuracy(

            name="masked_token_accuracy"
        )

    ]
)


model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=10,

    restore_best_weights=True,

    verbose=1
)


reduce_lr = ReduceLROnPlateau(

    monitor="val_loss",

    factor=0.5,

    patience=4,

    min_lr=1e-6,

    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

history = model.fit(

    X_train,

    y_train,

    sample_weight=train_weights,

    validation_data=(

        X_val,

        y_val,

        val_weights

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[

        early_stopping,
        reduce_lr

    ],

    verbose=1
)


# ============================================================
# EVALUATE
# ============================================================

results = model.evaluate(

    X_val,

    y_val,

    sample_weight=val_weights,

    verbose=0
)


print("\n===================================")
print("VALIDATION RESULTS")
print("===================================")

print(
    "Loss:",
    results[0]
)

print(
    "Masked Token Accuracy:",
    results[1]
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(

    MODEL_DIR
    /
    "tiny_bert_mlm.keras"
)


# ============================================================
# SAVE VOCABULARY
# ============================================================

with open(

    MODEL_DIR
    /
    "vocabulary.pkl",

    "wb"

) as file:

    pickle.dump(

        {

            "word_to_id":
                word_to_id,

            "id_to_word":
                id_to_word,

            "max_length":
                MAX_LENGTH

        },

        file
    )


# ============================================================
# FILL MASK FUNCTION
# ============================================================

def fill_mask(
    sentence,
    top_k=5
):

    # Allow user to use:
    #
    # _
    #
    # instead of explicitly typing [MASK]
    #
    sentence = sentence.replace(
        "_",
        MASK_TOKEN
    )


    # --------------------------------------------------------
    # Tokenize manually while preserving [MASK]
    # --------------------------------------------------------

    sentence = sentence.lower().strip()


    parts = re.findall(

        r"\[mask\]|[a-z0-9]+(?:'[a-z0-9]+)?",

        sentence
    )


    words = [

        MASK_TOKEN
        if word == "[mask]"
        else word

        for word in parts

    ]


    if MASK_TOKEN not in words:

        print(
            "Please include '_' or [MASK] in the sentence."
        )

        return None


    token_ids = [

        word_to_id.get(
            word,
            UNK_ID
        )

        for word in words

    ]


    token_ids = (

        [CLS_ID]
        +
        token_ids
        +
        [SEP_ID]

    )


    token_ids = token_ids[
        :MAX_LENGTH
    ]


    if token_ids[-1] != SEP_ID:

        token_ids[-1] = SEP_ID


    token_ids += (

        [PAD_ID]
        *
        (
            MAX_LENGTH
            -
            len(token_ids)
        )

    )


    input_array = np.array(

        [token_ids],

        dtype=np.int32
    )


    predictions = model.predict(

        input_array,

        verbose=0
    )


    mask_positions = [

        i

        for i, token_id
        in enumerate(token_ids)

        if token_id == MASK_ID

    ]


    if not mask_positions:

        return None


    # For this demo:
    # handle the first mask.
    mask_position = mask_positions[0]


    mask_logits = predictions[

        0,

        mask_position,

        :

    ]


    probabilities = tf.nn.softmax(

        mask_logits

    ).numpy()


    # Prevent special tokens being predictions
    for special_id in [

        PAD_ID,
        UNK_ID,
        MASK_ID,
        CLS_ID,
        SEP_ID

    ]:

        probabilities[
            special_id
        ] = 0.0


    top_indices = np.argsort(

        probabilities

    )[-top_k:][::-1]


    print(
        "\nTop predictions:"
    )


    for token_id in top_indices:

        print(

            f"{id_to_word[token_id]:20s}"
            f"{probabilities[token_id]:.4f}"

        )


    best_token_id = int(
        top_indices[0]
    )


    predicted_word = id_to_word[
        best_token_id
    ]


    # --------------------------------------------------------
    # Construct completed sentence
    # --------------------------------------------------------

    completed_words = words.copy()


    mask_word_index = (
        completed_words
        .index(
            MASK_TOKEN
        )
    )


    completed_words[
        mask_word_index
    ] = predicted_word


    completed_sentence = " ".join(
        completed_words
    )


    return completed_sentence


# ============================================================
# INTERACTIVE DEMO
# ============================================================

print("\n===================================")
print("BERT MASKED WORD PREDICTION")
print("===================================")

print(
    "Example: i _ playing when the rain started"
)

print(
    "Type 'quit' to stop."
)


while True:

    user_input = input(
        "\nSentence: "
    ).strip()


    if user_input.lower() == "quit":

        break


    output = fill_mask(

        user_input,

        top_k=5
    )


    if output:

        print(
            "\nCompleted:"
        )

        print(
            output
        )