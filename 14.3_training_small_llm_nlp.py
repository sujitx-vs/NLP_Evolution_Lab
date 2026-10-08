# ============================================================
# SMALL DECODER-ONLY LANGUAGE MODEL
# ============================================================
#
# Dataset:
#     D:\processed_corpus\documents.jsonl
#
# Corpus discovered:
#     158 documents
#     ~3.24M words
#     ~5.85M SentencePiece tokens
#
# GOAL:
#
#     Train a small decoder-only BASE language model
#     using the COMPLETE prepared corpus.
#
# IMPORTANT:
#
#     - No document is truncated.
#     - Large textbooks are fully used.
#     - Small documents are packed together.
#     - <EOS> separates documents.
#     - Images / Python files were already excluded.
#
#
# Architecture:
#
# Token IDs
#     ↓
# Shared trainable token embedding
# +
# learned positional embedding
#     ↓
# Decoder block × 2
#     ├── RMSNorm
#     ├── causal self-attention
#     ├── residual
#     ├── RMSNorm
#     ├── SwiGLU FFN
#     └── residual
#     ↓
# RMSNorm
#     ↓
# SAME token embedding matrix used as output projection
#     ↓
# vocabulary logits
#
#
# This is a BASE LANGUAGE MODEL.
#
# It will learn:
#
#     prompt → continuation
#
# Instruction tuning comes AFTER base pretraining.
# ============================================================


from pathlib import Path
import json
import math
import random
import re

import numpy as np
import tensorflow as tf
import keras

from keras import layers
from keras import ops

import sentencepiece as spm
from ftfy import fix_text


# ============================================================
# CONFIGURATION
# ============================================================

DOCUMENTS_FILE = Path(
    r"D:\processed_corpus\documents.jsonl"
)


OUTPUT_DIR = Path(
    r"D:\small_llm"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# TOKENIZER PATHS
# ============================================================

# IMPORTANT:
#
# We deliberately use V2.
#
# Your original tokenizer had already learned some PDF
# encoding corruption.
#
# We do NOT want to reuse:
#
# small_llm_tokenizer.model
#
# from the previous run.


TOKENIZER_CORPUS = (
    OUTPUT_DIR
    /
    "tokenizer_corpus_v2.txt"
)


TOKENIZER_PREFIX = (
    OUTPUT_DIR
    /
    "small_llm_tokenizer_v2"
)


TOKENIZER_MODEL = Path(
    str(TOKENIZER_PREFIX)
    +
    ".model"
)


# ============================================================
# TOKENIZER CONFIG
# ============================================================

VOCAB_SIZE = 4000


# ============================================================
# MODEL CONFIG
# ============================================================
#
# Actual corpus:
#
# ~5.85M tokens
#
# So we deliberately keep the network small.
#
# This is inspired by the Chinchilla lesson:
#
# DON'T make the model huge relative to the available data.
#
# This is NOT literally the Chinchilla architecture.
# ============================================================

CONTEXT_LENGTH = 256

D_MODEL = 48

NUM_HEADS = 4

NUM_LAYERS = 2

FF_DIM = 192

DROPOUT = 0.10


# ============================================================
# TRAINING CONFIG
# ============================================================

BATCH_SIZE = 32

EPOCHS = 10

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 0.01

RANDOM_SEED = 42


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(
    RANDOM_SEED
)


np.random.seed(
    RANDOM_SEED
)


tf.random.set_seed(
    RANDOM_SEED
)


# ============================================================
# GPU INFORMATION
# ============================================================

print(
    "\n==================================="
)

print(
    "HARDWARE"
)

print(
    "==================================="
)


gpus = tf.config.list_physical_devices(
    "GPU"
)


if gpus:

    print(
        f"GPU devices found: {len(gpus)}"
    )

    for gpu in gpus:

        print(
            gpu
        )

else:

    print(
        "No GPU detected. TensorFlow will use CPU."
    )


# ============================================================
# TEXT REPAIR
# ============================================================

def repair_text(text):
    """
    Fix common PDF encoding/mojibake problems without doing
    aggressive NLP preprocessing.

    We intentionally DO NOT:

    - lowercase everything
    - remove stopwords
    - stem
    - lemmatize
    - remove punctuation

    because this is language-model training.
    """

    if not text:

        return ""


    # --------------------------------------------------------
    # ftfy fixes many encoding issues such as mojibake.
    # --------------------------------------------------------

    text = fix_text(
        text
    )


    # --------------------------------------------------------
    # Remove null characters
    # --------------------------------------------------------

    text = text.replace(
        "\x00",
        " "
    )


    # --------------------------------------------------------
    # Normalize line endings
    # --------------------------------------------------------

    text = text.replace(
        "\r\n",
        "\n"
    )

    text = text.replace(
        "\r",
        "\n"
    )


    # --------------------------------------------------------
    # Excessive horizontal whitespace
    # --------------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )


    # --------------------------------------------------------
    # Excessive blank lines
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text.strip()


# ============================================================
# LOAD ALL DOCUMENTS
# ============================================================

def load_documents():

    documents = []


    with open(
        DOCUMENTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        for line in file:

            line = line.strip()


            if not line:

                continue


            item = json.loads(
                line
            )


            text = repair_text(
                item.get(
                    "text",
                    ""
                )
            )


            if not text:

                continue


            documents.append({

                "source":
                    item.get(
                        "source",
                        "unknown"
                    ),

                "text":
                    text

            })


    return documents


documents = load_documents()


print(
    "\n==================================="
)

print(
    "DOCUMENTS LOADED"
)

print(
    "==================================="
)


print(
    f"Documents: {len(documents)}"
)


if len(documents) == 0:

    raise RuntimeError(
        "No documents were loaded."
    )


# ============================================================
# CORPUS STATISTICS
# ============================================================

total_characters = sum(

    len(
        document["text"]
    )

    for document
    in documents

)


print(
    f"Characters after repair: "
    f"{total_characters:,}"
)


# ============================================================
# CREATE TOKENIZER CORPUS
# ============================================================
#
# ALL 158 DOCUMENTS ARE USED.
#
# No train/test document exclusion here.
#
# ============================================================

print(
    "\n==================================="
)

print(
    "PREPARING TOKENIZER CORPUS"
)

print(
    "==================================="
)


with open(
    TOKENIZER_CORPUS,
    "w",
    encoding="utf-8"
) as file:

    for index, document in enumerate(
        documents,
        start=1
    ):

        text = document[
            "text"
        ]


        for line in text.splitlines():

            line = line.strip()


            if line:

                file.write(
                    line
                    +
                    "\n"
                )


        if index % 20 == 0:

            print(
                f"Prepared "
                f"{index}/"
                f"{len(documents)} "
                f"documents"
            )


print(
    f"\nTokenizer corpus: "
    f"{TOKENIZER_CORPUS}"
)


# ============================================================
# TRAIN SENTENCEPIECE BPE TOKENIZER
# ============================================================
#
# We use a new V2 tokenizer so that the tokenizer from the
# previous run containing malformed PDF pieces is NOT reused.
# ============================================================

if not TOKENIZER_MODEL.exists():

    print(
        "\n==================================="
    )

    print(
        "TRAINING NEW V2 TOKENIZER"
    )

    print(
        "==================================="
    )


    spm.SentencePieceTrainer.train(

        input=str(
            TOKENIZER_CORPUS
        ),

        model_prefix=str(
            TOKENIZER_PREFIX
        ),

        vocab_size=VOCAB_SIZE,

        model_type="bpe",

        character_coverage=1.0,

        unk_id=0,

        bos_id=1,

        eos_id=2,

        pad_id=3,

        unk_piece="<unk>",

        bos_piece="<BOS>",

        eos_piece="<EOS>",

        pad_piece="<PAD>",

        hard_vocab_limit=False,

        # Avoid skipping unusually long extracted PDF lines
        max_sentence_length=16384

    )


else:

    print(
        "\nExisting V2 tokenizer found."
    )

    print(
        "Skipping tokenizer training."
    )


# ============================================================
# LOAD TOKENIZER
# ============================================================

tokenizer = (
    spm.SentencePieceProcessor()
)


success = tokenizer.load(
    str(
        TOKENIZER_MODEL
    )
)


if not success:

    raise RuntimeError(
        "Failed to load SentencePiece tokenizer."
    )


ACTUAL_VOCAB_SIZE = (
    tokenizer.get_piece_size()
)


UNK_ID = tokenizer.unk_id()

BOS_ID = tokenizer.bos_id()

EOS_ID = tokenizer.eos_id()

PAD_ID = tokenizer.pad_id()


print(
    "\n==================================="
)

print(
    "TOKENIZER"
)

print(
    "==================================="
)


print(
    f"Vocabulary size: "
    f"{ACTUAL_VOCAB_SIZE:,}"
)


print(
    f"UNK ID: {UNK_ID}"
)

print(
    f"BOS ID: {BOS_ID}"
)

print(
    f"EOS ID: {EOS_ID}"
)

print(
    f"PAD ID: {PAD_ID}"
)


# ============================================================
# TOKENIZER SANITY CHECK
# ============================================================

sample_text = (
    "Artificial intelligence uses machine learning "
    "to solve complex problems."
)


sample_pieces = tokenizer.encode(

    sample_text,

    out_type=str

)


print(
    "\nTokenizer sample:"
)

print(
    sample_pieces
)


# ============================================================
# CREATE COMPLETE TOKEN STREAM
# ============================================================
#
# OPTION B PACKING
#
# Document A tokens
# <EOS>
# Document B tokens
# <EOS>
# Document C tokens
# <EOS>
#
#
# IMPORTANT:
#
# No book is shortened.
#
# If a book contains 1 million tokens, all one million tokens
# enter this stream.
#
# Small documents naturally share 256-token training windows.
# ============================================================

def create_complete_token_stream(
    document_list
):

    token_chunks = []

    total_tokens = 0


    print(
        "\n==================================="
    )

    print(
        "TOKENIZING COMPLETE CORPUS"
    )

    print(
        "==================================="
    )


    for index, document in enumerate(
        document_list,
        start=1
    ):

        token_ids = tokenizer.encode(

            document["text"],

            out_type=int

        )


        # ----------------------------------------------------
        # Entire document retained.
        # ----------------------------------------------------

        document_array = np.asarray(

            token_ids
            +
            [EOS_ID],

            dtype=np.int32

        )


        token_chunks.append(
            document_array
        )


        total_tokens += len(
            document_array
        )


        if (
            index % 10 == 0
            or
            index == len(
                document_list
            )
        ):

            print(

                f"Tokenized "
                f"{index}/"
                f"{len(document_list)} "
                f"documents | "
                f"{total_tokens:,} tokens"

            )


    # --------------------------------------------------------
    # Concatenate AFTER each document was individually
    # tokenized.
    # --------------------------------------------------------

    complete_stream = np.concatenate(
        token_chunks
    )


    return complete_stream


all_tokens = create_complete_token_stream(
    documents
)


TOTAL_TOKENS = len(
    all_tokens
)


print(
    "\n==================================="
)

print(
    "CORPUS TOKEN COUNT"
)

print(
    "==================================="
)


print(
    f"Total training tokens: "
    f"{TOTAL_TOKENS:,}"
)


# ============================================================
# SAVE TOKEN IDS
# ============================================================
#
# This means future runs don't have to tokenize 158 documents
# again if you later separate training from preprocessing.
# ============================================================

TOKEN_IDS_FILE = (
    OUTPUT_DIR
    /
    "all_corpus_tokens_v2.npy"
)


np.save(
    TOKEN_IDS_FILE,
    all_tokens
)


print(
    f"Saved token IDs: "
    f"{TOKEN_IDS_FILE}"
)


# ============================================================
# CREATE TRAINING DATASET
# ============================================================
#
# Example sequence:
#
# t1 t2 t3 t4 t5
#
# input:
#
# t1 t2 t3 t4
#
# target:
#
# t2 t3 t4 t5
#
#
# We need 257 tokens to construct:
#
# 256 input tokens
# 256 target tokens
#
#
# Every batch updates the SAME model parameters.
# ============================================================

def create_lm_dataset(
    token_stream
):

    dataset = (
        tf.data.Dataset
        .from_tensor_slices(
            token_stream
        )
    )


    # --------------------------------------------------------
    # 257 token chunks
    # --------------------------------------------------------

    dataset = dataset.batch(

        CONTEXT_LENGTH + 1,

        drop_remainder=True

    )


    # --------------------------------------------------------
    # Shift input/target by one
    # --------------------------------------------------------

    def split_input_target(
        sequence
    ):

        inputs = sequence[
            :-1
        ]


        targets = sequence[
            1:
        ]


        return (
            inputs,
            targets
        )


    dataset = dataset.map(

        split_input_target,

        num_parallel_calls=
            tf.data.AUTOTUNE

    )


    # --------------------------------------------------------
    # Shuffle sequences, NOT individual tokens.
    # --------------------------------------------------------

    dataset = dataset.shuffle(

        buffer_size=8192,

        seed=RANDOM_SEED,

        reshuffle_each_iteration=True

    )


    # --------------------------------------------------------
    # Training batches
    # --------------------------------------------------------

    dataset = dataset.batch(

        BATCH_SIZE,

        drop_remainder=False

    )


    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )


    return dataset


train_dataset = create_lm_dataset(
    all_tokens
)


# ============================================================
# DATASET STATISTICS
# ============================================================

number_sequences = (

    TOTAL_TOKENS
    //
    (
        CONTEXT_LENGTH
        +
        1
    )

)


steps_per_epoch = math.ceil(

    number_sequences
    /
    BATCH_SIZE

)


print(
    "\n==================================="
)

print(
    "TRAINING DATASET"
)

print(
    "==================================="
)


print(
    f"Context length: "
    f"{CONTEXT_LENGTH}"
)


print(
    f"Training sequences: "
    f"{number_sequences:,}"
)


print(
    f"Batch size: "
    f"{BATCH_SIZE}"
)


print(
    f"Approx steps/epoch: "
    f"{steps_per_epoch:,}"
)


print(
    f"Epochs: "
    f"{EPOCHS}"
)


# ============================================================
# RMS NORMALIZATION
# ============================================================

@keras.utils.register_keras_serializable()
class RMSNorm(
    layers.Layer
):

    def __init__(
        self,
        epsilon=1e-6,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


        self.epsilon = epsilon


    def build(
        self,
        input_shape
    ):

        self.scale = self.add_weight(

            name="scale",

            shape=(
                input_shape[-1],
            ),

            initializer="ones",

            trainable=True

        )


        super().build(
            input_shape
        )


    def call(
        self,
        inputs
    ):

        rms = ops.sqrt(

            ops.mean(

                ops.square(
                    inputs
                ),

                axis=-1,

                keepdims=True

            )

            +
            self.epsilon

        )


        return (

            inputs
            /
            rms

        ) * self.scale


    def get_config(
        self
    ):

        config = (
            super()
            .get_config()
        )


        config.update({

            "epsilon":
                self.epsilon

        })


        return config


# ============================================================
# TIED TOKEN EMBEDDING
# ============================================================
#
# SAME MATRIX USED FOR:
#
# 1. Input token embeddings
#
#        token ID → vector
#
# 2. Output vocabulary logits
#
#        hidden vector → vocabulary
#
#
# This reduces parameters significantly.
# ============================================================

@keras.utils.register_keras_serializable()
class TiedTokenEmbedding(
    layers.Layer
):

    def __init__(
        self,
        vocab_size,
        d_model,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


        self.vocab_size = (
            vocab_size
        )


        self.d_model = (
            d_model
        )


    def build(
        self,
        input_shape
    ):

        self.embedding_matrix = (
            self.add_weight(

                name="embedding_matrix",

                shape=(

                    self.vocab_size,

                    self.d_model

                ),

                initializer=
                    keras.initializers
                    .RandomNormal(
                        stddev=0.02
                    ),

                trainable=True

            )
        )


        super().build(
            input_shape
        )


    def call(
        self,
        inputs,
        mode="embed"
    ):

        if mode == "embed":

            return ops.take(

                self.embedding_matrix,

                inputs,

                axis=0

            )


        if mode == "project":

            return ops.matmul(

                inputs,

                ops.transpose(
                    self.embedding_matrix
                )

            )


        raise ValueError(
            "mode must be 'embed' or 'project'"
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

            "d_model":
                self.d_model

        })


        return config


# ============================================================
# POSITION EMBEDDING
# ============================================================

@keras.utils.register_keras_serializable()
class LearnedPositionEmbedding(
    layers.Layer
):

    def __init__(
        self,
        max_length,
        d_model,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


        self.max_length = (
            max_length
        )


        self.d_model = (
            d_model
        )


        self.position_embedding = (
            layers.Embedding(

                input_dim=max_length,

                output_dim=d_model

            )
        )


    def call(
        self,
        token_vectors
    ):

        sequence_length = (
            ops.shape(
                token_vectors
            )[1]
        )


        positions = ops.arange(
            sequence_length
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

            "max_length":
                self.max_length,

            "d_model":
                self.d_model

        })


        return config


# ============================================================
# SWIGLU FEED FORWARD NETWORK
# ============================================================

@keras.utils.register_keras_serializable()
class SwiGLU(
    layers.Layer
):

    def __init__(
        self,
        ff_dim,
        d_model,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


        self.ff_dim = (
            ff_dim
        )


        self.d_model = (
            d_model
        )


        # FIX:
        #
        # Always use layers.Dense
        #
        # not unqualified Dense(...)

        self.gate_projection = (
            layers.Dense(

                ff_dim,

                use_bias=False

            )
        )


        self.value_projection = (
            layers.Dense(

                ff_dim,

                use_bias=False

            )
        )


        self.output_projection = (
            layers.Dense(

                d_model,

                use_bias=False

            )
        )


    def call(
        self,
        inputs
    ):

        gate = self.gate_projection(
            inputs
        )


        value = self.value_projection(
            inputs
        )


        # SiLU(x)
        activated_gate = (

            gate
            *
            ops.sigmoid(
                gate
            )

        )


        hidden = (

            activated_gate
            *
            value

        )


        return self.output_projection(
            hidden
        )


    def get_config(
        self
    ):

        config = (
            super()
            .get_config()
        )


        config.update({

            "ff_dim":
                self.ff_dim,

            "d_model":
                self.d_model

        })


        return config


# ============================================================
# TRANSFORMER DECODER BLOCK
# ============================================================

@keras.utils.register_keras_serializable()
class DecoderBlock(
    layers.Layer
):

    def __init__(
        self,
        d_model,
        num_heads,
        ff_dim,
        dropout_rate,
        **kwargs
    ):

        super().__init__(
            **kwargs
        )


        self.d_model = (
            d_model
        )


        self.num_heads = (
            num_heads
        )


        self.ff_dim = (
            ff_dim
        )


        self.dropout_rate = (
            dropout_rate
        )


        if (
            d_model
            %
            num_heads
            !=
            0
        ):

            raise ValueError(

                "D_MODEL must be divisible "
                "by NUM_HEADS."

            )


        self.head_dimension = (

            d_model
            //
            num_heads

        )


        # ----------------------------------------------------
        # PRE-NORMALIZATION
        # ----------------------------------------------------

        self.norm1 = RMSNorm()

        self.norm2 = RMSNorm()


        # ----------------------------------------------------
        # CAUSAL SELF-ATTENTION
        # ----------------------------------------------------

        self.attention = (
            layers.MultiHeadAttention(

                num_heads=
                    num_heads,

                key_dim=
                    self.head_dimension,

                dropout=
                    dropout_rate,

                use_bias=False

            )
        )


        # ----------------------------------------------------
        # SWIGLU
        # ----------------------------------------------------

        self.ffn = SwiGLU(

            ff_dim=
                ff_dim,

            d_model=
                d_model

        )


        # FIX:
        #
        # Use layers.Dropout everywhere.

        self.dropout1 = (
            layers.Dropout(
                dropout_rate
            )
        )


        self.dropout2 = (
            layers.Dropout(
                dropout_rate
            )
        )


    def call(
        self,
        inputs,
        training=False
    ):

        # ====================================================
        # ATTENTION
        # ====================================================

        normalized = self.norm1(
            inputs
        )


        attention_output = (
            self.attention(

                query=
                    normalized,

                key=
                    normalized,

                value=
                    normalized,

                use_causal_mask=True,

                training=
                    training

            )
        )


        attention_output = (
            self.dropout1(

                attention_output,

                training=
                    training

            )
        )


        x = (

            inputs
            +
            attention_output

        )


        # ====================================================
        # FEED FORWARD NETWORK
        # ====================================================

        normalized = self.norm2(
            x
        )


        ffn_output = self.ffn(
            normalized
        )


        ffn_output = (
            self.dropout2(

                ffn_output,

                training=
                    training

            )
        )


        x = (

            x
            +
            ffn_output

        )


        return x


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
#
# IMPORTANT CHANGE:
#
# shape=(None,)
#
# instead of:
#
# shape=(CONTEXT_LENGTH,)
#
#
# Training sequences are still 256 tokens.
#
# But during generation we can send:
#
# 5 tokens
# 20 tokens
# 100 tokens
# ...
#
# without adding PAD tokens.
# ============================================================

input_ids = keras.Input(

    shape=(
        None,
    ),

    dtype="int32",

    name="input_ids"

)


# ============================================================
# SHARED TOKEN EMBEDDING
# ============================================================

token_embedding = TiedTokenEmbedding(

    vocab_size=
        ACTUAL_VOCAB_SIZE,

    d_model=
        D_MODEL,

    name=
        "shared_token_embedding"

)


x = token_embedding(

    input_ids,

    mode="embed"

)


# ============================================================
# POSITIONAL INFORMATION
# ============================================================

x = LearnedPositionEmbedding(

    max_length=
        CONTEXT_LENGTH,

    d_model=
        D_MODEL,

    name=
        "position_embedding"

)(x)


# ============================================================
# EMBEDDING DROPOUT
# ============================================================

x = layers.Dropout(

    DROPOUT,

    name=
        "embedding_dropout"

)(x)


# ============================================================
# TRANSFORMER BLOCKS
# ============================================================

for layer_number in range(
    NUM_LAYERS
):

    x = DecoderBlock(

        d_model=
            D_MODEL,

        num_heads=
            NUM_HEADS,

        ff_dim=
            FF_DIM,

        dropout_rate=
            DROPOUT,

        name=(
            f"decoder_block_"
            f"{layer_number + 1}"
        )

    )(x)


# ============================================================
# FINAL NORMALIZATION
# ============================================================

x = RMSNorm(

    name=
        "final_norm"

)(x)


# ============================================================
# TIED OUTPUT PROJECTION
# ============================================================
#
# NO separate Dense(VOCAB_SIZE).
#
# Instead:
#
# hidden states
#       ×
# transpose(token embedding matrix)
#
# This is TRUE weight tying.
# ============================================================

logits = token_embedding(

    x,

    mode="project"

)


# ============================================================
# CREATE MODEL
# ============================================================

model = keras.Model(

    inputs=
        input_ids,

    outputs=
        logits,

    name=
        "small_document_language_model"

)


# ============================================================
# MODEL SUMMARY
# ============================================================

model.summary()


PARAMETER_COUNT = (
    model.count_params()
)


# ============================================================
# DATA / MODEL SCALE
# ============================================================

tokens_per_parameter = (

    TOTAL_TOKENS
    /
    PARAMETER_COUNT

)


print(
    "\n==================================="
)

print(
    "MODEL / DATA SCALE"
)

print(
    "==================================="
)


print(
    f"Parameters: "
    f"{PARAMETER_COUNT:,}"
)


print(
    f"Unique training tokens: "
    f"{TOTAL_TOKENS:,}"
)


print(
    f"Tokens per parameter: "
    f"{tokens_per_parameter:.2f}"
)


print(
    "\nNOTE:"
)

print(
    "This ratio uses UNIQUE corpus tokens "
    "from one pass through the dataset."
)


print(
    "Repeated epochs do not create new "
    "independent training data."
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = keras.optimizers.AdamW(

    learning_rate=
        LEARNING_RATE,

    weight_decay=
        WEIGHT_DECAY,

    beta_1=
        0.9,

    beta_2=
        0.95,

    epsilon=
        1e-8,

    clipnorm=
        1.0

)


# ============================================================
# LOSS
# ============================================================

loss_function = (
    keras.losses
    .SparseCategoricalCrossentropy(

        from_logits=True

    )
)


# ============================================================
# COMPILE MODEL
# ============================================================

model.compile(

    optimizer=
        optimizer,

    loss=
        loss_function,

    metrics=[

        keras.metrics
        .SparseCategoricalAccuracy(

            name=
                "token_accuracy"

        )

    ]

)


# ============================================================
# CALLBACKS
# ============================================================
#
# Since ALL documents are used for training,
# we do NOT pretend to have a true validation set.
#
# We monitor training loss.
#
# Later, if we want scientifically independent evaluation,
# we'll create a separate unseen corpus.
# ============================================================

BEST_MODEL_PATH = (

    OUTPUT_DIR
    /
    "best_base_model_v2.keras"

)


callbacks = [

    keras.callbacks.ModelCheckpoint(

        filepath=str(
            BEST_MODEL_PATH
        ),

        monitor=
            "loss",

        mode=
            "min",

        save_best_only=
            True,

        verbose=
            1

    ),

    keras.callbacks.TerminateOnNaN()

]


# ============================================================
# START PRETRAINING
# ============================================================

print(
    "\n==================================="
)

print(
    "STARTING BASE MODEL PRETRAINING"
)

print(
    "==================================="
)


print(
    "Every prepared document participates "
    "in training."
)


print(
    "No textbook is truncated."
)


history = model.fit(

    train_dataset,

    epochs=
        EPOCHS,

    callbacks=
        callbacks,

    verbose=
        1

)


# ============================================================
# SAVE FINAL MODEL
# ============================================================

FINAL_MODEL_PATH = (

    OUTPUT_DIR
    /
    "final_base_model_v2.keras"

)


model.save(
    FINAL_MODEL_PATH
)


print(
    "\n==================================="
)

print(
    "TRAINING COMPLETE"
)

print(
    "==================================="
)


print(
    f"Best model:\n"
    f"{BEST_MODEL_PATH}"
)


print(
    f"\nFinal model:\n"
    f"{FINAL_MODEL_PATH}"
)


# ============================================================
# SAVE TRAINING CONFIGURATION
# ============================================================

config = {

    "documents":
        len(
            documents
        ),

    "total_tokens":
        int(
            TOTAL_TOKENS
        ),

    "vocab_size":
        int(
            ACTUAL_VOCAB_SIZE
        ),

    "context_length":
        CONTEXT_LENGTH,

    "d_model":
        D_MODEL,

    "num_heads":
        NUM_HEADS,

    "num_layers":
        NUM_LAYERS,

    "ff_dim":
        FF_DIM,

    "dropout":
        DROPOUT,

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE,

    "parameter_count":
        int(
            PARAMETER_COUNT
        ),

    "tokens_per_parameter":
        float(
            tokens_per_parameter
        )

}


CONFIG_FILE = (

    OUTPUT_DIR
    /
    "training_config_v2.json"

)


with open(

    CONFIG_FILE,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        config,

        file,

        indent=4

    )


# ============================================================
# TEXT GENERATION
# ============================================================
#
# Important:
#
# NO LEFT PADDING.
#
# The model accepts variable sequence lengths.
#
# If prompt is:
#
#     "Machine learning is"
#
# and tokenization produces 5 tokens,
#
# the model sees exactly those 5 tokens.
#
# ============================================================

def generate_text(
    prompt,
    max_new_tokens=120,
    temperature=0.8,
    top_k=40
):

    if temperature <= 0:

        raise ValueError(
            "temperature must be > 0"
        )


    token_ids = tokenizer.encode(

        prompt,

        out_type=int

    )


    # If completely empty prompt
    if not token_ids:

        token_ids = [
            BOS_ID
        ]


    generated_ids = list(
        token_ids
    )


    for _ in range(
        max_new_tokens
    ):

        # ----------------------------------------------------
        # Only last CONTEXT_LENGTH tokens are visible.
        # ----------------------------------------------------

        context = generated_ids[
            -CONTEXT_LENGTH:
        ]


        model_input = np.asarray(

            [
                context
            ],

            dtype=np.int32

        )


        predictions = model(

            model_input,

            training=False

        )


        # ----------------------------------------------------
        # Last REAL token position
        # ----------------------------------------------------

        next_logits = predictions[

            0,

            -1,

            :

        ]


        # ----------------------------------------------------
        # Prevent generation of PAD/BOS
        # ----------------------------------------------------

        next_logits = tf.tensor_scatter_nd_update(

            next_logits,

            indices=[

                [PAD_ID],

                [BOS_ID]

            ],

            updates=[

                tf.constant(
                    -1e9,
                    dtype=
                        next_logits.dtype
                ),

                tf.constant(
                    -1e9,
                    dtype=
                        next_logits.dtype
                )

            ]

        )


        # ----------------------------------------------------
        # TEMPERATURE
        # ----------------------------------------------------

        next_logits = (

            next_logits
            /
            temperature

        )


        # ----------------------------------------------------
        # TOP-K
        # ----------------------------------------------------

        k = min(

            top_k,

            ACTUAL_VOCAB_SIZE

        )


        top_values, top_indices = (
            tf.math.top_k(

                next_logits,

                k=k

            )
        )


        # ----------------------------------------------------
        # SAMPLE ONE TOKEN
        # ----------------------------------------------------

        sampled_index = (
            tf.random.categorical(

                logits=
                    tf.expand_dims(
                        top_values,
                        axis=0
                    ),

                num_samples=
                    1

            )[0, 0]
        )


        next_token = int(

            top_indices[
                sampled_index
            ].numpy()

        )


        # ----------------------------------------------------
        # EOS = stop generation
        # ----------------------------------------------------

        if next_token == EOS_ID:

            break


        generated_ids.append(
            next_token
        )


    return tokenizer.decode(
        generated_ids
    )


# ============================================================
# NEXT TOKEN INSPECTION
# ============================================================

def show_next_token_predictions(
    prompt,
    top_k=10
):

    token_ids = tokenizer.encode(

        prompt,

        out_type=int

    )


    if not token_ids:

        token_ids = [
            BOS_ID
        ]


    context = token_ids[
        -CONTEXT_LENGTH:
    ]


    model_input = np.asarray(

        [
            context
        ],

        dtype=np.int32

    )


    predictions = model(

        model_input,

        training=False

    )


    logits = predictions[
        0,
        -1,
        :
    ]


    probabilities = (
        tf.nn.softmax(
            logits
        )
        .numpy()
    )


    top_ids = np.argsort(
        probabilities
    )[-top_k:][::-1]


    print(
        "\nTop next-token predictions:"
    )


    for token_id in top_ids:

        piece = tokenizer.id_to_piece(
            int(
                token_id
            )
        )


        print(

            f"{piece:25s} "
            f"{probabilities[token_id]:.4f}"

        )


# ============================================================
# INTERACTIVE TEST
# ============================================================

print(
    "\n==================================="
)

print(
    "BASE LANGUAGE MODEL TEST"
)

print(
    "==================================="
)


print(
    "This is a BASE model."
)


print(
    "It performs continuation, not instruction following yet."
)


print(
    "\nType 'quit' to exit."
)


while True:

    prompt = input(
        "\nPrompt: "
    ).strip()


    if prompt.lower() == "quit":

        break


    if not prompt:

        continue


    show_next_token_predictions(

        prompt,

        top_k=5

    )


    output = generate_text(

        prompt=
            prompt,

        max_new_tokens=
            120,

        temperature=
            0.8,

        top_k=
            40

    )


    print(
        "\nGenerated:\n"
    )


    print(
        output
    )