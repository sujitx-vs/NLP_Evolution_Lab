# ============================================================
# TINY DECODER-ONLY TRANSFORMER LANGUAGE MODEL
# ============================================================
#
# Dataset:
# datasets/LLM_Foundations.pdf
#
# Tokenization:
# SentencePiece BPE subword tokenizer
#
# Goal:
# Learn next-token prediction from a book/document.
#
# Architecture:
#
# PDF
#  ↓
# Text Extraction
#  ↓
# Text Cleaning
#  ↓
# BPE Subword Tokenization
#  ↓
# Token IDs
#  ↓
# Token + Positional Embeddings
#  ↓
# Causal Multi-Head Self-Attention
#  ↓
# FFN
#  ↓
# Transformer Blocks
#  ↓
# Vocabulary Projection
#  ↓
# Logits
#  ↓
# Next-token probabilities
#
# ============================================================


import re
import json
import unicodedata
from pathlib import Path

import numpy as np
import tensorflow as tf
import keras
import sentencepiece as spm

from pypdf import PdfReader

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


# ============================================================
# PATHS
# ============================================================

PDF_PATH = Path(
    "datasets/LLM_Foundations.pdf"
)


MODEL_DIR = Path(
    "models/tiny_decoder_transformer_bpe"
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


CORPUS_DIR = Path(
    "datasets/processed_lm"
)


CORPUS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


TRAIN_TEXT_PATH = (
    CORPUS_DIR
    /
    "llm_foundations_train.txt"
)


VAL_TEXT_PATH = (
    CORPUS_DIR
    /
    "llm_foundations_validation.txt"
)


TOKENIZER_PREFIX = (
    MODEL_DIR
    /
    "bpe_tokenizer"
)


TOKENIZER_MODEL = Path(
    str(TOKENIZER_PREFIX)
    +
    ".model"
)


# ============================================================
# MODEL CONFIGURATION
# ============================================================

CONTEXT_LENGTH = 64

D_MODEL = 64

NUM_HEADS = 4

FF_DIM = 128

NUM_LAYERS = 2

DROPOUT_RATE = 0.1


# ============================================================
# TOKENIZER CONFIGURATION
# ============================================================

BPE_VOCAB_SIZE = 500


# ============================================================
# DATA WINDOW CONFIGURATION
# ============================================================
#
# Previous experiment:
#
# window 1: token 1 -> 32
# window 2: token 2 -> 33
# window 3: token 3 -> 34
#
# For a whole book this creates huge duplication.
#
# Here:
#
# move 16 tokens at a time.
#
# Every window still predicts EVERY next token inside
# the window because of causal training.
#
# ============================================================

WINDOW_STRIDE = 16


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

BATCH_SIZE = 32

EPOCHS = 100

LEARNING_RATE = 0.0005

TRAIN_RATIO = 0.90


# ============================================================
# RANDOM SEEDS
# ============================================================

np.random.seed(42)

tf.random.set_seed(42)


# ============================================================
# CHECK PDF
# ============================================================

if not PDF_PATH.exists():

    raise FileNotFoundError(
        f"PDF not found: {PDF_PATH}"
    )


# ============================================================
# PDF TEXT EXTRACTION
# ============================================================

def extract_pdf_pages(pdf_path):

    reader = PdfReader(
        str(pdf_path)
    )

    pages = []


    print("\n===================================")
    print("PDF EXTRACTION")
    print("===================================")


    print(
        "Total PDF pages:",
        len(reader.pages)
    )


    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            page_text = (
                page.extract_text()
                or ""
            )

        except Exception as error:

            print(
                f"Warning: Page {page_number} "
                f"could not be extracted:",
                error
            )

            page_text = ""


        pages.append(
            page_text
        )


    return pages


raw_pages = extract_pdf_pages(
    PDF_PATH
)


# ============================================================
# CHECK WHETHER PDF CONTAINS REAL TEXT
# ============================================================

total_extracted_characters = sum(

    len(page)

    for page in raw_pages
)


print(
    "Extracted characters:",
    total_extracted_characters
)


if total_extracted_characters < 1000:

    raise ValueError(

        "\nVery little text was extracted from the PDF.\n"
        "The PDF may be scanned/image-based.\n"
        "OCR would be required before training."
    )


# ============================================================
# TEXT CLEANING
# ============================================================
#
# IMPORTANT:
#
# We DO NOT aggressively remove punctuation anymore.
#
# SentencePiece can learn punctuation as tokens/subwords.
#
#
# Main cleaning:
#
# 1. Unicode normalization
# 2. Remove null characters
# 3. Fix words broken by PDF line wrapping:
#
#       transfor-
#       mer
#
#       ->
#
#       transformer
#
# 4. Remove standalone page numbers
# 5. Reduce repeated spaces
# 6. Preserve normal punctuation
#
# ============================================================

def clean_pdf_text(text):

    # Unicode normalization
    text = unicodedata.normalize(
        "NFKC",
        text
    )


    # Remove NULL characters
    text = text.replace(
        "\x00",
        ""
    )


    # --------------------------------------------
    # Fix hyphenated words broken across lines
    #
    # transfor-
    # mer
    #
    # -> transformer
    # --------------------------------------------

    text = re.sub(

        r"([A-Za-z])-\s*\n\s*([A-Za-z])",

        r"\1\2",

        text
    )


    # --------------------------------------------
    # Remove standalone page numbers
    #
    # Example:
    #
    # 42
    #
    # --------------------------------------------

    text = re.sub(

        r"(?m)^\s*\d+\s*$",

        " ",

        text
    )


    # --------------------------------------------
    # Convert line breaks inside normal paragraphs
    # into spaces.
    #
    # Keep paragraph boundaries.
    # --------------------------------------------

    text = re.sub(

        r"(?<!\n)\n(?!\n)",

        " ",

        text
    )


    # --------------------------------------------
    # Normalize repeated spaces/tabs
    # --------------------------------------------

    text = re.sub(

        r"[ \t]+",

        " ",

        text
    )


    # --------------------------------------------
    # Normalize huge blank areas
    # --------------------------------------------

    text = re.sub(

        r"\n{3,}",

        "\n\n",

        text
    )


    return text.strip()


clean_pages = [

    clean_pdf_text(page)

    for page in raw_pages

    if page.strip()
]


print(
    "Usable text pages:",
    len(clean_pages)
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================
#
# IMPORTANT:
#
# We split BEFORE training the tokenizer.
#
# This prevents the tokenizer from learning directly from
# validation text.
#
# Earlier:
#
# tokenize everything
#      ↓
# split token stream
#
#
# Here:
#
# extracted pages
#      ↓
# train pages / validation pages
#      ↓
# tokenizer trained ONLY on training text
#
# ============================================================

split_page = int(

    len(clean_pages)
    *
    TRAIN_RATIO
)


train_pages = clean_pages[
    :split_page
]


val_pages = clean_pages[
    split_page:
]


train_text = "\n\n".join(
    train_pages
)


val_text = "\n\n".join(
    val_pages
)


print("\n===================================")
print("DATA SPLIT")
print("===================================")


print(
    "Training pages:",
    len(train_pages)
)


print(
    "Validation pages:",
    len(val_pages)
)


print(
    "Training characters:",
    len(train_text)
)


print(
    "Validation characters:",
    len(val_text)
)


# ============================================================
# SAVE CLEAN CORPUS
# ============================================================
#
# SentencePiece trains from text files.
#
# ============================================================

TRAIN_TEXT_PATH.write_text(

    train_text,

    encoding="utf-8"
)


VAL_TEXT_PATH.write_text(

    val_text,

    encoding="utf-8"
)


# ============================================================
# TRAIN BPE TOKENIZER
# ============================================================
#
# BPE:
#
# transformer
#
# might become:
#
# ▁transform
# er
#
#
# SentencePiece's ▁ symbol represents a word boundary /
# preceding whitespace.
#
#
# IMPORTANT:
#
# hard_vocab_limit=False
#
# prevents training from failing if this book does not
# contain enough unique subword combinations to create
# exactly 2000 tokens.
#
# ============================================================

if not TOKENIZER_MODEL.exists():

    print("\n===================================")
    print("TRAINING BPE TOKENIZER")
    print("===================================")


    spm.SentencePieceTrainer.train(

        input=str(
            TRAIN_TEXT_PATH
        ),

        model_prefix=str(
            TOKENIZER_PREFIX
        ),

        vocab_size=BPE_VOCAB_SIZE,

        model_type="bpe",

        character_coverage=1.0,

        unk_id=0,

        bos_id=1,

        eos_id=2,

        pad_id=-1,

        hard_vocab_limit=False
    )


else:

    print(
        "\nExisting tokenizer found:",
        TOKENIZER_MODEL
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = spm.SentencePieceProcessor(

    model_file=str(
        TOKENIZER_MODEL
    )
)


VOCAB_SIZE = tokenizer.vocab_size()


print("\n===================================")
print("BPE TOKENIZER")
print("===================================")


print(
    "Vocabulary size:",
    VOCAB_SIZE
)


# ============================================================
# TOKENIZER DEMONSTRATION
# ============================================================

sample_sentence = (
    "Transformer architectures use "
    "self attention for sequence processing."
)


sample_pieces = tokenizer.encode(

    sample_sentence,

    out_type=str
)


sample_ids = tokenizer.encode(

    sample_sentence,

    out_type=int
)


print(
    "\nExample sentence:"
)


print(
    sample_sentence
)


print(
    "\nSubword pieces:"
)


print(
    sample_pieces
)


print(
    "\nToken IDs:"
)


print(
    sample_ids
)


print(
    "\nDecoded again:"
)


print(
    tokenizer.decode(
        sample_ids
    )
)


# ============================================================
# TEXT -> SUBWORD TOKEN IDS
# ============================================================

train_token_ids = tokenizer.encode(

    train_text,

    out_type=int
)


val_token_ids = tokenizer.encode(

    val_text,

    out_type=int
)


print("\n===================================")
print("TOKEN COUNTS")
print("===================================")


print(
    "Training tokens:",
    len(train_token_ids)
)


print(
    "Validation tokens:",
    len(val_token_ids)
)


# ============================================================
# SHOW FIRST SUBWORD TOKENS
# ============================================================

print(
    "\nFirst 40 BPE pieces:"
)


print(

    [

        tokenizer.id_to_piece(
            token_id
        )

        for token_id
        in train_token_ids[:40]

    ]

)


# ============================================================
# CREATE LANGUAGE-MODEL WINDOWS
# ============================================================
#
# Example:
#
# token stream:
#
# t1 t2 t3 t4 t5
#
#
# Input:
#
# t1 t2 t3 t4
#
#
# Target:
#
# t2 t3 t4 t5
#
#
# Causal masking means:
#
# t1          -> t2
# t1 t2       -> t3
# t1 t2 t3    -> t4
# t1 t2 t3 t4 -> t5
#
# ============================================================

def create_sequences(

    tokens,

    context_length,

    stride
):

    X = []

    y = []


    for i in range(

        0,

        len(tokens)
        -
        context_length
        -
        1,

        stride
    ):


        sequence = tokens[

            i:
            i
            +
            context_length
            +
            1

        ]


        X.append(

            sequence[:-1]

        )


        y.append(

            sequence[1:]

        )


    return (

        np.asarray(
            X,
            dtype=np.int32
        ),

        np.asarray(
            y,
            dtype=np.int32
        )

    )


X_train, y_train = create_sequences(

    train_token_ids,

    CONTEXT_LENGTH,

    WINDOW_STRIDE
)


X_val, y_val = create_sequences(

    val_token_ids,

    CONTEXT_LENGTH,

    WINDOW_STRIDE
)


print("\n===================================")
print("TRAINING WINDOWS")
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
    "X_val:",
    X_val.shape
)


print(
    "y_val:",
    y_val.shape
)


if len(X_train) == 0:

    raise ValueError(

        "Not enough training tokens for the "
        "selected CONTEXT_LENGTH."
    )


if len(X_val) == 0:

    raise ValueError(

        "Not enough validation tokens for the "
        "selected CONTEXT_LENGTH."
    )


# ============================================================
# SHOW ONE TRAINING SAMPLE
# ============================================================

print("\n===================================")
print("EXAMPLE TRAINING SAMPLE")
print("===================================")


print(
    "\nINPUT TOKEN PIECES:\n"
)


print(

    [

        tokenizer.id_to_piece(
            int(token_id)
        )

        for token_id
        in X_train[0]

    ]

)


print(
    "\nINPUT DECODED:\n"
)


print(

    tokenizer.decode(

        X_train[0].tolist()

    )

)


print(
    "\nTARGET DECODED:\n"
)


print(

    tokenizer.decode(

        y_train[0].tolist()

    )

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


        self.vocab_size = vocab_size

        self.max_length = max_length

        self.d_model = d_model


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
# DECODER-ONLY TRANSFORMER BLOCK
# ============================================================
#
# NO ENCODER
# NO CROSS-ATTENTION
#
# Only:
#
# causal self-attention
# +
# FFN
#
# ============================================================

@keras.utils.register_keras_serializable()
class DecoderBlock(
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

        self.dropout_rate = dropout_rate


        # ============================================
        # CAUSAL MULTI-HEAD SELF-ATTENTION
        # ============================================

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


        # ============================================
        # FEED-FORWARD NETWORK
        # ============================================

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

        # ============================================
        # PRE-NORM
        # ============================================

        x_norm = self.norm1(
            inputs
        )


        # ============================================
        # CAUSAL SELF-ATTENTION
        #
        # query = current sequence
        # key   = current sequence
        # value = current sequence
        #
        # use_causal_mask=True
        #
        # prevents looking into future tokens.
        # ============================================

        attention_output = (

            self.self_attention(

                query=x_norm,

                key=x_norm,

                value=x_norm,

                use_causal_mask=True,

                training=training
            )

        )


        attention_output = (

            self.dropout1(

                attention_output,

                training=training
            )

        )


        # ============================================
        # FIRST RESIDUAL
        # ============================================

        x = (

            inputs
            +
            attention_output

        )


        # ============================================
        # SECOND PRE-NORM
        # ============================================

        x_norm2 = self.norm2(
            x
        )


        # ============================================
        # FFN
        # ============================================

        ffn_output = self.ffn(
            x_norm2
        )


        ffn_output = (

            self.dropout2(

                ffn_output,

                training=training
            )

        )


        # ============================================
        # SECOND RESIDUAL
        # ============================================

        return (

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

    shape=(None,),

    dtype="int32",

    name="token_ids"
)


# ============================================================
# TOKEN + POSITION EMBEDDINGS
# ============================================================

x = TokenAndPositionEmbedding(

    vocab_size=VOCAB_SIZE,

    max_length=CONTEXT_LENGTH,

    d_model=D_MODEL,

    name="token_position_embedding"

)(inputs)


x = Dropout(
    DROPOUT_RATE
)(x)


# ============================================================
# DECODER BLOCKS
# ============================================================

for i in range(
    NUM_LAYERS
):

    x = DecoderBlock(

        d_model=D_MODEL,

        num_heads=NUM_HEADS,

        ff_dim=FF_DIM,

        dropout_rate=DROPOUT_RATE,

        name=f"decoder_block_{i + 1}"

    )(x)


# ============================================================
# FINAL LAYER NORMALIZATION
# ============================================================

x = LayerNormalization(

    epsilon=1e-6,

    name="final_layer_norm"

)(x)


# ============================================================
# VOCABULARY PROJECTION
# ============================================================
#
# d_model
#
# 64
# ↓
# Dense
# ↓
# BPE vocabulary
#
# Example:
#
# 2000 logits
#
# ============================================================

logits = Dense(

    VOCAB_SIZE,

    name="vocabulary_projection"

)(x)


# ============================================================
# MODEL
# ============================================================

model = Model(

    inputs=inputs,

    outputs=logits,

    name="tiny_decoder_only_transformer_bpe"
)


# ============================================================
# COMPILE
# ============================================================

optimizer = keras.optimizers.Adam(

    learning_rate=LEARNING_RATE
)


loss_function = (

    keras.losses
    .SparseCategoricalCrossentropy(

        from_logits=True
    )

)


model.compile(

    optimizer=optimizer,

    loss=loss_function,

    metrics=[

        keras.metrics
        .SparseCategoricalAccuracy(

            name="token_accuracy"
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

    validation_data=(

        X_val,

        y_val

    ),

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    callbacks=[

        # early_stopping,

        # reduce_lr

    ],

    verbose=1
)


# ============================================================
# VALIDATION
# ============================================================

validation_results = model.evaluate(

    X_val,

    y_val,

    verbose=0
)


validation_loss = validation_results[0]

validation_accuracy = validation_results[1]


# ============================================================
# PERPLEXITY
# ============================================================
#
# Perplexity = exp(cross entropy loss)
#
# ============================================================

perplexity = np.exp(
    validation_loss
)


print("\n===================================")
print("VALIDATION RESULTS")
print("===================================")


print(
    "Loss:",
    validation_loss
)


print(
    "Perplexity:",
    perplexity
)


print(
    "Token Accuracy:",
    validation_accuracy
)


# ============================================================
# NEXT SUBWORD TOKEN PREDICTION
# ============================================================

def predict_next_tokens(

    prompt,

    top_k=10
):

    token_ids = tokenizer.encode(

        prompt,

        out_type=int
    )


    if not token_ids:

        return []


    # --------------------------------------------
    # Keep only available context
    # --------------------------------------------

    token_ids = token_ids[
        -CONTEXT_LENGTH:
    ]


    input_array = np.array(

        [token_ids],

        dtype=np.int32
    )


    predictions = model.predict(

        input_array,

        verbose=0
    )


    # Last position
    next_token_logits = predictions[

        0,

        -1,

        :

    ]


    probabilities = tf.nn.softmax(

        next_token_logits

    ).numpy()


    top_indices = np.argsort(

        probabilities

    )[-top_k:][::-1]


    results = []


    for token_id in top_indices:

        piece = tokenizer.id_to_piece(

            int(token_id)
        )


        probability = float(

            probabilities[
                token_id
            ]
        )


        results.append(

            (
                int(token_id),
                piece,
                probability
            )

        )


    return results


# ============================================================
# TOP-K SAMPLING
# ============================================================
#
# Instead of always using:
#
# argmax()
#
# we sample among the top candidate tokens.
#
# This helps reduce deterministic repetition loops.
#
# ============================================================

def sample_next_token(

    logits,

    temperature=0.8,

    top_k=20
):

    logits = np.asarray(
        logits,
        dtype=np.float64
    )


    # --------------------------------------------
    # TEMPERATURE
    # --------------------------------------------

    logits = logits / max(
        temperature,
        1e-6
    )


    # --------------------------------------------
    # TOP-K FILTER
    # --------------------------------------------

    top_k = min(

        top_k,

        len(logits)
    )


    top_indices = np.argpartition(

        logits,

        -top_k

    )[-top_k:]


    top_logits = logits[
        top_indices
    ]


    # Numerical stability
    top_logits = (

        top_logits

        -

        np.max(
            top_logits
        )

    )


    probabilities = np.exp(
        top_logits
    )


    probabilities = (

        probabilities

        /

        probabilities.sum()

    )


    selected_index = np.random.choice(

        len(top_indices),

        p=probabilities
    )


    return int(

        top_indices[
            selected_index
        ]

    )


# ============================================================
# GENERATE TEXT
# ============================================================

def generate_text(

    prompt,

    max_new_tokens=50,

    temperature=0.8,

    top_k=20
):

    generated_ids = tokenizer.encode(

        prompt,

        out_type=int
    )


    if not generated_ids:

        return ""


    eos_id = tokenizer.eos_id()


    for _ in range(
        max_new_tokens
    ):


        context_ids = generated_ids[

            -CONTEXT_LENGTH:

        ]


        input_array = np.array(

            [context_ids],

            dtype=np.int32
        )


        predictions = model.predict(

            input_array,

            verbose=0

        )


        next_token_logits = predictions[

            0,

            -1,

            :

        ]


        next_token_id = sample_next_token(

            next_token_logits,

            temperature=temperature,

            top_k=top_k
        )


        # --------------------------------------------
        # EOS stopping condition
        # --------------------------------------------

        if (

            eos_id >= 0

            and

            next_token_id == eos_id

        ):

            break


        generated_ids.append(
            next_token_id
        )


    # --------------------------------------------
    # SentencePiece joins subwords automatically
    # --------------------------------------------

    return tokenizer.decode(
        generated_ids
    )


# ============================================================
# SAVE MODEL
# ============================================================

model.save(

    MODEL_DIR
    /
    "tiny_decoder_transformer_bpe.keras"
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {

    "pdf":
        str(PDF_PATH),

    "context_length":
        CONTEXT_LENGTH,

    "window_stride":
        WINDOW_STRIDE,

    "vocab_size":
        VOCAB_SIZE,

    "d_model":
        D_MODEL,

    "num_heads":
        NUM_HEADS,

    "ff_dim":
        FF_DIM,

    "num_layers":
        NUM_LAYERS,

    "tokenizer":
        "SentencePiece BPE",

    "tokenizer_model":
        str(TOKENIZER_MODEL)

}


with open(

    MODEL_DIR
    /
    "config.json",

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        config,

        file,

        indent=4
    )


print(
    "\nModel saved to:",
    MODEL_DIR
)


print(
    "Tokenizer saved to:",
    TOKENIZER_MODEL
)


# ============================================================
# INTERACTIVE TEST
# ============================================================

print("\n===================================")
print("NEXT SUBWORD-TOKEN PREDICTION")
print("===================================")


print(
    "Type 'quit' to stop."
)


while True:

    prompt = input(
        "\nPrompt: "
    ).strip()


    if prompt.lower() == "quit":

        break


    predictions = predict_next_tokens(

        prompt,

        top_k=10
    )


    print(
        "\nTop next-token predictions:"
    )


    for (

        token_id,

        piece,

        probability

    ) in predictions:


        print(

            f"{token_id:5d}  "
            f"{piece:20s}  "
            f"{probability:.4f}"

        )


    print(
        "\nGenerated continuation:"
    )


    generated = generate_text(

        prompt,

        max_new_tokens=50,

        temperature=0.8,

        top_k=20
    )


    print(
        generated
    )