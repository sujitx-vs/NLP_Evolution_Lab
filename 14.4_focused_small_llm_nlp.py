# ============================================================
# FOCUSED SMALL LANGUAGE MODEL
# ============================================================
#
# Original corpus:
#     D:\processed_corpus\documents.jsonl
#
# Focus:
#     AI / ML / Deep Learning / NLP
#     Data Science / Analytics
#     Python
#     Java / OOP / Software topics
#
# Excluded:
#     Aptitude / reasoning
#     Grammar / vocabulary books
#     Assignments / MCQs / question sheets
#     Syllabi
#
#
# IMPORTANT:
#
# Once a document is selected:
#
#     THE COMPLETE DOCUMENT IS USED.
#
# No:
#     5000-word limit
#     page truncation
#     book truncation
#
#
# Architecture:
#
# Focused technical corpus
#       ↓
# SentencePiece BPE
#       ↓
# Trainable tied embeddings
#       +
# learned positional embeddings
#       ↓
# Decoder Transformer x 3
#       ├── RMSNorm
#       ├── causal self-attention
#       ├── residual
#       ├── RMSNorm
#       ├── SwiGLU
#       └── residual
#       ↓
# RMSNorm
#       ↓
# tied vocabulary projection
#       ↓
# next-token logits
#
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
# PATHS
# ============================================================

DOCUMENTS_FILE = Path(
    r"D:\processed_corpus\documents.jsonl"
)


OUTPUT_DIR = Path(
    r"D:\focused_slm"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


FOCUSED_DOCUMENTS_FILE = (
    OUTPUT_DIR
    /
    "focused_documents.jsonl"
)


SELECTION_REPORT_FILE = (
    OUTPUT_DIR
    /
    "focused_selection_report.json"
)


TOKENIZER_CORPUS = (
    OUTPUT_DIR
    /
    "focused_tokenizer_corpus.txt"
)


TOKENIZER_PREFIX = (
    OUTPUT_DIR
    /
    "focused_slm_tokenizer"
)


TOKENIZER_MODEL = Path(
    str(TOKENIZER_PREFIX)
    +
    ".model"
)


TOKEN_IDS_FILE = (
    OUTPUT_DIR
    /
    "focused_tokens.npy"
)


# ============================================================
# SOURCE SELECTION
# ============================================================
#
# These are matched against NORMALIZED filenames.
#
# Example:
#
# "Hands-On_Machine_Learning..."
#
# becomes approximately:
#
# "hands on machine learning ..."
#
# ============================================================


INCLUDE_PATTERNS = [

    # --------------------------------------------------------
    # ARTIFICIAL INTELLIGENCE
    # --------------------------------------------------------

    "artificial intelligence",
    "fundamentals of ai",
    "fundementals of ai",
    "mathematics for ai",
    "math ai",
    "ai by examples",
    "ai compute",
    "basic understanding of deep neural",
    "neural network",
    "deep neural",
    "dnn",

    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    "machine learning",
    "scikit learn",
    "clustering",
    "bias variance",
    "hyperparameter",
    "hyperparater",
    "optimizers",
    "gradient",
    "batchnormalization",
    "batch normalization",
    "dropout",
    "earlystopping",
    "early stopping",

    # --------------------------------------------------------
    # DEEP LEARNING
    # --------------------------------------------------------

    "cnn",
    "rnn",
    "gan",
    "activation",
    "loss",

    # --------------------------------------------------------
    # NLP / TRANSFORMERS
    # --------------------------------------------------------

    "nlp",
    "natural language",
    "transformer",
    "bert",
    "word2vec",
    "glove",
    "shallowparsing",
    "shallow parsing",
    "language in cognitive",

    # --------------------------------------------------------
    # DATA SCIENCE / ANALYTICS
    # --------------------------------------------------------

    "data science",
    "datascience",
    "data analytics",
    "business analytics",
    "data analysis",
    "data engineering",
    "dataengg",

    # --------------------------------------------------------
    # PYTHON
    # --------------------------------------------------------

    "python",
    "pandas",

    # --------------------------------------------------------
    # JAVA / SOFTWARE DEVELOPMENT
    # --------------------------------------------------------

    "java",
    "oop",
    "ooad",
    "object oriented",
    "spring boot",
    "nodejs",
    "functional programming"

]


# ============================================================
# STRONG EXCLUSION RULES
# ============================================================
#
# Exclusion ALWAYS wins over inclusion.
#
# Example:
#
# AI_Assignment1.pdf
#
# contains "AI"
#
# but also contains "assignment"
#
# therefore:
#
# EXCLUDED
# ============================================================

EXCLUDE_PATTERNS = [

    # --------------------------------------------------------
    # ASSIGNMENTS
    # --------------------------------------------------------

    "assignment",
    "assign01",
    "assign02",
    "assign03",
    "assign04",
    "assign05",
    "assign06",
    "assign07",
    "assign08",
    "assign09",
    "assign10",
    "homework",
    "miniproject",
    "mini project",

    # --------------------------------------------------------
    # EXAMS / QUESTION SHEETS
    # --------------------------------------------------------

    "mcq",
    "poll questions",
    "mock",
    "question paper",
    "problemstatement",
    "problem statement",

    # --------------------------------------------------------
    # SYLLABUS
    # --------------------------------------------------------

    "syllabus",

    # --------------------------------------------------------
    # APTITUDE / REASONING
    # --------------------------------------------------------

    "aptitude",
    "apti",
    "reasoning",
    "ratio questions",
    "quantitative aptitude",

    # --------------------------------------------------------
    # GRAMMAR / VOCABULARY
    # --------------------------------------------------------

    "english grammar",
    "idioms",
    "phrases",
    "synonyms",
    "antonyms",
    "word power",
    "prepositions",

    # --------------------------------------------------------
    # RANDOM / ADMINISTRATIVE
    # --------------------------------------------------------

    "meeting"

]


# ============================================================
# TOKENIZER CONFIGURATION
# ============================================================
#
# Focused corpus is smaller than the mixed corpus.
#
# A 3000-token vocabulary keeps:
#
# - vocabulary projection small
# - embeddings compact
# - subword coverage strong
#
# ============================================================

VOCAB_SIZE = 3000


# ============================================================
# MODEL CONFIGURATION
# ============================================================
#
# This is stronger than the previous tiny model,
# but still intentionally small.
#
# Approximate target:
#
# ~400K–600K parameters
#
# Actual number will be printed.
#
# ============================================================

CONTEXT_LENGTH = 256

D_MODEL = 64

NUM_HEADS = 4

NUM_LAYERS = 3

FF_DIM = 256

DROPOUT = 0.10


# ============================================================
# TRAINING CONFIG
# ============================================================

BATCH_SIZE = 32

EPOCHS = 15

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
# HARDWARE
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
# NORMALIZE FILE NAME
# ============================================================

def normalize_filename(
    filename
):

    filename = filename.lower()


    # Remove extension
    filename = re.sub(
        r"\.[^.]+$",
        "",
        filename
    )


    # Replace separators with spaces
    filename = re.sub(
        r"[_\-]+",
        " ",
        filename
    )


    # Remove duplicated whitespace
    filename = re.sub(
        r"\s+",
        " ",
        filename
    )


    return filename.strip()


# ============================================================
# CHECK PATTERN
# ============================================================

def contains_pattern(
    normalized_name,
    pattern
):

    pattern = pattern.lower().strip()

    return pattern in normalized_name


# ============================================================
# DECIDE WHETHER DOCUMENT IS FOCUSED
# ============================================================

def document_is_selected(
    source
):

    normalized = normalize_filename(
        source
    )


    # --------------------------------------------------------
    # EXCLUSION HAS PRIORITY
    # --------------------------------------------------------

    for pattern in EXCLUDE_PATTERNS:

        if contains_pattern(
            normalized,
            pattern
        ):

            return (
                False,
                f"excluded:{pattern}"
            )


    # --------------------------------------------------------
    # THEN CHECK TARGET TOPICS
    # --------------------------------------------------------

    for pattern in INCLUDE_PATTERNS:

        if contains_pattern(
            normalized,
            pattern
        ):

            return (
                True,
                f"included:{pattern}"
            )


    return (
        False,
        "no_target_keyword"
    )


# ============================================================
# TEXT REPAIR
# ============================================================

def repair_text(
    text
):

    if not text:

        return ""


    # --------------------------------------------------------
    # Repair encoding problems
    # --------------------------------------------------------

    text = fix_text(
        text
    )


    # --------------------------------------------------------
    # Remove nulls
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
    # Remove huge repeated separators
    #
    # Example:
    #
    # ________________________________
    #
    # -------------------------------
    #
    # These were being learned by BPE previously.
    # --------------------------------------------------------

    text = re.sub(
        r"[_]{5,}",
        " ",
        text
    )


    text = re.sub(
        r"[-]{8,}",
        " ",
        text
    )


    text = re.sub(
        r"[=]{8,}",
        " ",
        text
    )


    # --------------------------------------------------------
    # Normalize spaces
    # --------------------------------------------------------

    text = re.sub(
        r"[ \t]+",
        " ",
        text
    )


    # --------------------------------------------------------
    # Normalize blank lines
    # --------------------------------------------------------

    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )


    return text.strip()


# ============================================================
# LOAD AND FILTER DOCUMENTS
# ============================================================

def load_focused_documents():

    selected = []

    excluded = []


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


            source = item.get(
                "source",
                "unknown"
            )


            include, reason = (
                document_is_selected(
                    source
                )
            )


            if not include:

                excluded.append({

                    "source":
                        source,

                    "reason":
                        reason

                })

                continue


            text = repair_text(

                item.get(
                    "text",
                    ""
                )

            )


            if not text:

                excluded.append({

                    "source":
                        source,

                    "reason":
                        "empty_after_cleaning"

                })

                continue


            selected.append({

                "source":
                    source,

                "type":
                    item.get(
                        "type",
                        ""
                    ),

                "text":
                    text,

                "characters":
                    len(
                        text
                    ),

                "words":
                    len(
                        re.findall(
                            r"\b\w+\b",
                            text
                        )
                    ),

                "selection_reason":
                    reason

            })


    return (
        selected,
        excluded
    )


documents, excluded_documents = (
    load_focused_documents()
)


# ============================================================
# VERIFY CORPUS
# ============================================================

if len(documents) == 0:

    raise RuntimeError(
        "Focused corpus contains zero documents."
    )


# ============================================================
# CORPUS STATISTICS
# ============================================================

total_characters = sum(

    document[
        "characters"
    ]

    for document
    in documents

)


total_words = sum(

    document[
        "words"
    ]

    for document
    in documents

)


print(
    "\n==================================="
)

print(
    "FOCUSED CORPUS"
)

print(
    "==================================="
)


print(
    f"Selected documents: "
    f"{len(documents)}"
)


print(
    f"Excluded documents: "
    f"{len(excluded_documents)}"
)


print(
    f"Focused words: "
    f"{total_words:,}"
)


print(
    f"Focused characters: "
    f"{total_characters:,}"
)


# ============================================================
# PRINT SELECTED FILES
# ============================================================

print(
    "\n==================================="
)

print(
    "SELECTED SOURCES"
)

print(
    "==================================="
)


for index, document in enumerate(
    documents,
    start=1
):

    print(

        f"{index:3d}. "
        f"{document['source']} "
        f"| {document['words']:,} words"

    )


# ============================================================
# SAVE FOCUSED DOCUMENTS
# ============================================================

with open(

    FOCUSED_DOCUMENTS_FILE,

    "w",

    encoding="utf-8"

) as file:

    for document in documents:

        file.write(

            json.dumps(

                document,

                ensure_ascii=False

            )

            +
            "\n"

        )


# ============================================================
# SAVE SELECTION REPORT
# ============================================================

selection_report = {

    "selected_document_count":
        len(
            documents
        ),

    "excluded_document_count":
        len(
            excluded_documents
        ),

    "total_words":
        total_words,

    "total_characters":
        total_characters,

    "selected_sources": [

        document[
            "source"
        ]

        for document
        in documents

    ],

    "excluded_sources":
        excluded_documents

}


with open(

    SELECTION_REPORT_FILE,

    "w",

    encoding="utf-8"

) as file:

    json.dump(

        selection_report,

        file,

        indent=4,

        ensure_ascii=False

    )


# ============================================================
# PREPARE TOKENIZER CORPUS
# ============================================================

print(
    "\n==================================="
)

print(
    "PREPARING FOCUSED TOKENIZER CORPUS"
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


        if (
            index % 10 == 0
            or
            index == len(
                documents
            )
        ):

            print(

                f"Prepared "
                f"{index}/"
                f"{len(documents)}"

            )


# ============================================================
# TRAIN NEW TOKENIZER
# ============================================================
#
# IMPORTANT:
#
# Do NOT reuse tokenizer from mixed-corpus model.
#
# This tokenizer should learn vocabulary from:
#
# AI / ML / NLP / programming text
#
# rather than:
#
# aptitude / grammar / reasoning books.
# ============================================================

if not TOKENIZER_MODEL.exists():

    print(
        "\n==================================="
    )

    print(
        "TRAINING FOCUSED TOKENIZER"
    )

    print(
        "==================================="
    )


    spm.SentencePieceTrainer.train(

        input=
            str(
                TOKENIZER_CORPUS
            ),

        model_prefix=
            str(
                TOKENIZER_PREFIX
            ),

        vocab_size=
            VOCAB_SIZE,

        model_type=
            "bpe",

        character_coverage=
            1.0,

        unk_id=
            0,

        bos_id=
            1,

        eos_id=
            2,

        pad_id=
            3,

        unk_piece=
            "<unk>",

        bos_piece=
            "<BOS>",

        eos_piece=
            "<EOS>",

        pad_piece=
            "<PAD>",

        hard_vocab_limit=
            False,

        max_sentence_length=
            16384

    )


else:

    print(
        "\nFocused tokenizer already exists."
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
        "Could not load focused tokenizer."
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
    "TOKENIZER INFORMATION"
)

print(
    "==================================="
)


print(
    f"Vocabulary: "
    f"{ACTUAL_VOCAB_SIZE:,}"
)


print(
    f"BOS: {BOS_ID}"
)


print(
    f"EOS: {EOS_ID}"
)


print(
    f"PAD: {PAD_ID}"
)


# ============================================================
# TOKENIZER SANITY TEST
# ============================================================

test_sentences = [

    "Artificial intelligence uses machine learning.",

    "A transformer uses self attention.",

    "Java supports object oriented programming.",

    "Python is commonly used for data science."

]


print(
    "\nTokenizer examples:"
)


for sentence in test_sentences:

    print(
        "\nTEXT:"
    )

    print(
        sentence
    )


    print(
        "TOKENS:"
    )

    print(

        tokenizer.encode(

            sentence,

            out_type=str

        )

    )


# ============================================================
# TOKENIZE ALL SELECTED DOCUMENTS
# ============================================================
#
# OPTION B:
#
# document 1
# <EOS>
# document 2
# <EOS>
# document 3
#
#
# Complete books remain complete.
# ============================================================

def create_complete_token_stream(
    document_list
):

    arrays = []

    token_count = 0


    print(
        "\n==================================="
    )

    print(
        "TOKENIZING FOCUSED CORPUS"
    )

    print(
        "==================================="
    )


    for index, document in enumerate(
        document_list,
        start=1
    ):

        tokens = tokenizer.encode(

            document[
                "text"
            ],

            out_type=int

        )


        # ----------------------------------------------------
        # FULL DOCUMENT + EOS
        # ----------------------------------------------------

        document_tokens = np.asarray(

            tokens
            +
            [
                EOS_ID
            ],

            dtype=np.int32

        )


        arrays.append(
            document_tokens
        )


        token_count += len(
            document_tokens
        )


        if (
            index % 10 == 0
            or
            index == len(
                document_list
            )
        ):

            print(

                f"{index}/"
                f"{len(document_list)} "
                f"documents | "
                f"{token_count:,} tokens"

            )


    return np.concatenate(
        arrays
    )


all_tokens = (
    create_complete_token_stream(
        documents
    )
)


TOTAL_TOKENS = len(
    all_tokens
)


print(
    "\n==================================="
)

print(
    "FOCUSED TOKEN COUNT"
)

print(
    "==================================="
)


print(
    f"Total tokens: "
    f"{TOTAL_TOKENS:,}"
)


# ============================================================
# SAVE TOKEN IDS
# ============================================================

np.save(

    TOKEN_IDS_FILE,

    all_tokens

)


print(
    f"Saved tokens to:\n"
    f"{TOKEN_IDS_FILE}"
)


# ============================================================
# CREATE LANGUAGE MODEL DATASET
# ============================================================
#
# We use:
#
# window length = 257
# shift         = 256
#
#
# Example:
#
# Window 1:
#
# tokens 0 ... 256
#
#
# Window 2:
#
# tokens 256 ... 512
#
#
# Therefore the boundary transition is preserved.
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


    dataset = dataset.window(

        size=
            CONTEXT_LENGTH + 1,

        shift=
            CONTEXT_LENGTH,

        drop_remainder=
            True

    )


    dataset = dataset.flat_map(

        lambda window:

            window.batch(
                CONTEXT_LENGTH + 1
            )

    )


    def split_sequence(
        sequence
    ):

        x = sequence[
            :-1
        ]


        y = sequence[
            1:
        ]


        return (
            x,
            y
        )


    dataset = dataset.map(

        split_sequence,

        num_parallel_calls=
            tf.data.AUTOTUNE

    )


    # --------------------------------------------------------
    # Shuffle SEQUENCES.
    #
    # Never shuffle individual tokens.
    # --------------------------------------------------------

    dataset = dataset.shuffle(

        buffer_size=
            8192,

        seed=
            RANDOM_SEED,

        reshuffle_each_iteration=
            True

    )


    dataset = dataset.batch(

        BATCH_SIZE,

        drop_remainder=
            False

    )


    dataset = dataset.prefetch(
        tf.data.AUTOTUNE
    )


    return dataset


train_dataset = create_lm_dataset(
    all_tokens
)


# ============================================================
# TRAINING STATISTICS
# ============================================================

number_sequences = max(

    0,

    (
        TOTAL_TOKENS
        -
        1
    )
    //
    CONTEXT_LENGTH

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
    "TRAINING DATA"
)

print(
    "==================================="
)


print(
    f"Sequences: "
    f"{number_sequences:,}"
)


print(
    f"Context length: "
    f"{CONTEXT_LENGTH}"
)


print(
    f"Batch size: "
    f"{BATCH_SIZE}"
)


print(
    f"Steps / epoch: "
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


        self.epsilon = (
            epsilon
        )


    def build(
        self,
        input_shape
    ):

        self.scale = (
            self.add_weight(

                name=
                    "scale",

                shape=(
                    input_shape[-1],
                ),

                initializer=
                    "ones",

                trainable=
                    True

            )
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
# TIED TOKEN EMBEDDINGS
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

                name=
                    "embedding_matrix",

                shape=(

                    self.vocab_size,

                    self.d_model

                ),

                initializer=
                    keras.initializers
                    .RandomNormal(
                        stddev=0.02
                    ),

                trainable=
                    True

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
            "mode must be embed or project"
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


        self.embedding = (
            layers.Embedding(

                input_dim=
                    max_length,

                output_dim=
                    d_model

            )
        )


    def call(
        self,
        token_vectors
    ):

        length = ops.shape(
            token_vectors
        )[1]


        positions = ops.arange(
            length
        )


        return (

            token_vectors

            +

            self.embedding(
                positions
            )

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
# SWIGLU
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


        self.gate = layers.Dense(

            ff_dim,

            use_bias=False

        )


        self.value = layers.Dense(

            ff_dim,

            use_bias=False

        )


        self.output_layer = (
            layers.Dense(

                d_model,

                use_bias=False

            )
        )


    def call(
        self,
        inputs
    ):

        gate = self.gate(
            inputs
        )


        value = self.value(
            inputs
        )


        swish_gate = (

            gate

            *

            ops.sigmoid(
                gate
            )

        )


        return self.output_layer(

            swish_gate
            *
            value

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
# DECODER BLOCK
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


        head_dim = (

            d_model
            //
            num_heads

        )


        self.norm1 = RMSNorm()

        self.norm2 = RMSNorm()


        self.attention = (
            layers.MultiHeadAttention(

                num_heads=
                    num_heads,

                key_dim=
                    head_dim,

                dropout=
                    dropout_rate,

                use_bias=
                    False

            )
        )


        self.ffn = SwiGLU(

            ff_dim=
                ff_dim,

            d_model=
                d_model

        )


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

        # ----------------------------------------------------
        # CAUSAL SELF ATTENTION
        # ----------------------------------------------------

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

                use_causal_mask=
                    True,

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


        # ----------------------------------------------------
        # FEED FORWARD
        # ----------------------------------------------------

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

input_ids = keras.Input(

    shape=(
        None,
    ),

    dtype=
        "int32",

    name=
        "input_ids"

)


# ============================================================
# TOKEN EMBEDDING
# ============================================================

token_embedding = (
    TiedTokenEmbedding(

        vocab_size=
            ACTUAL_VOCAB_SIZE,

        d_model=
            D_MODEL,

        name=
            "shared_token_embedding"

    )
)


x = token_embedding(

    input_ids,

    mode=
        "embed"

)


# ============================================================
# POSITION EMBEDDING
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
# DROPOUT
# ============================================================

x = layers.Dropout(

    DROPOUT,

    name=
        "embedding_dropout"

)(x)


# ============================================================
# TRANSFORMER STACK
# ============================================================

for i in range(
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

        name=
            f"decoder_block_{i + 1}"

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

logits = token_embedding(

    x,

    mode=
        "project"

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
        "focused_technical_slm"

)


model.summary()


# ============================================================
# MODEL STATISTICS
# ============================================================

PARAMETER_COUNT = (
    model.count_params()
)


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
    f"Training tokens: "
    f"{TOTAL_TOKENS:,}"
)


print(
    f"Tokens / parameter: "
    f"{tokens_per_parameter:.2f}"
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
# COMPILE
# ============================================================

model.compile(

    optimizer=
        optimizer,

    loss=
        keras.losses
        .SparseCategoricalCrossentropy(

            from_logits=
                True

        ),

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

BEST_MODEL_PATH = (

    OUTPUT_DIR
    /
    "best_focused_slm.keras"

)


callbacks = [

    keras.callbacks.ModelCheckpoint(

        filepath=
            str(
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

    keras.callbacks.ReduceLROnPlateau(

        monitor=
            "loss",

        factor=
            0.5,

        patience=
            2,

        min_lr=
            1e-6,

        verbose=
            1

    ),

    keras.callbacks.TerminateOnNaN()

]


# ============================================================
# TRAIN
# ============================================================

print(
    "\n==================================="
)

print(
    "TRAINING FOCUSED SLM"
)

print(
    "==================================="
)


print(
    f"Documents: "
    f"{len(documents)}"
)


print(
    f"Tokens: "
    f"{TOTAL_TOKENS:,}"
)


print(
    f"Parameters: "
    f"{PARAMETER_COUNT:,}"
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
    "final_focused_slm.keras"

)


model.save(
    FINAL_MODEL_PATH
)


# ============================================================
# SAVE CONFIG
# ============================================================

configuration = {

    "model_type":
        "focused_decoder_only_slm",

    "documents":
        len(
            documents
        ),

    "words":
        int(
            total_words
        ),

    "tokens":
        int(
            TOTAL_TOKENS
        ),

    "parameters":
        int(
            PARAMETER_COUNT
        ),

    "tokens_per_parameter":
        float(
            tokens_per_parameter
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

    "batch_size":
        BATCH_SIZE,

    "epochs":
        EPOCHS,

    "learning_rate":
        LEARNING_RATE

}


with open(

    OUTPUT_DIR
    /
    "focused_training_config.json",

    "w",

    encoding=
        "utf-8"

) as file:

    json.dump(

        configuration,

        file,

        indent=
            4

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
# GENERATION
# ============================================================

def apply_repetition_penalty(
    logits,
    generated_ids,
    penalty=1.10
):

    if penalty <= 1:

        return logits


    unique_tokens = set(
        generated_ids[
            -64:
        ]
    )


    logits_numpy = (
        logits.numpy()
        .copy()
    )


    for token_id in unique_tokens:

        if (
            token_id < 0
            or
            token_id >= len(
                logits_numpy
            )
        ):

            continue


        if (
            logits_numpy[
                token_id
            ]
            >
            0
        ):

            logits_numpy[
                token_id
            ] /= penalty

        else:

            logits_numpy[
                token_id
            ] *= penalty


    return tf.convert_to_tensor(

        logits_numpy,

        dtype=
            logits.dtype

    )


# ============================================================
# TEXT GENERATION
# ============================================================

def generate_text(
    prompt,
    max_new_tokens=100,
    temperature=0.70,
    top_k=20,
    repetition_penalty=1.10
):

    if temperature <= 0:

        raise ValueError(
            "temperature must be > 0"
        )


    token_ids = tokenizer.encode(

        prompt,

        out_type=int

    )


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


        next_logits = predictions[

            0,

            -1,

            :

        ]


        # ----------------------------------------------------
        # NEVER GENERATE PAD OR BOS
        # ----------------------------------------------------

        next_logits = tf.tensor_scatter_nd_update(

            next_logits,

            indices=[

                [
                    PAD_ID
                ],

                [
                    BOS_ID
                ]

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
        # REPETITION PENALTY
        # ----------------------------------------------------

        next_logits = (
            apply_repetition_penalty(

                next_logits,

                generated_ids,

                penalty=
                    repetition_penalty

            )
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

                k=
                    k

            )
        )


        sampled = (
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
                sampled
            ].numpy()

        )


        if next_token == EOS_ID:

            break


        generated_ids.append(
            next_token
        )


    return tokenizer.decode(
        generated_ids
    )


# ============================================================
# NEXT TOKEN DISTRIBUTION
# ============================================================

def show_next_token_predictions(
    prompt,
    top_k=5
):

    tokens = tokenizer.encode(

        prompt,

        out_type=int

    )


    if not tokens:

        tokens = [
            BOS_ID
        ]


    context = tokens[
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

            f"{piece:25s}"
            f"{probabilities[token_id]:.4f}"

        )


# ============================================================
# INTERACTIVE TEST
# ============================================================

print(
    "\n==================================="
)

print(
    "FOCUSED TECHNICAL SLM TEST"
)

print(
    "==================================="
)


print(
    "Still a BASE model — continuation only."
)


print(
    "Type 'quit' to exit."
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

        top_k=
            5

    )


    generated = generate_text(

        prompt=
            prompt,

        max_new_tokens=
            100,

        temperature=
            0.70,

        top_k=
            20,

        repetition_penalty=
            1.10

    )


    print(
        "\nGenerated:\n"
    )


    print(
        generated
    )