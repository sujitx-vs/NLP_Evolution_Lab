from pathlib import Path
import json
import pandas as pd


DOCUMENTS_FILE = Path(
    r"D:\processed_corpus\documents.jsonl"
)

REPORT_OUTPUT = Path(
    r"D:\processed_corpus\corpus_analysis.csv"
)


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

        documents.append(
            json.loads(line)
        )


df = pd.DataFrame(documents)


# ============================================================
# BASIC STATISTICS
# ============================================================

total_documents = len(df)

total_words = df["words"].sum()

total_characters = df["characters"].sum()


print("\n===================================")
print("CORPUS ANALYSIS")
print("===================================")

print(
    f"Documents: {total_documents}"
)

print(
    f"Total words: {total_words:,}"
)

print(
    f"Total characters: {total_characters:,}"
)


print("\n===================================")
print("DOCUMENT SIZE STATISTICS")
print("===================================")

print(
    f"Average words/document: "
    f"{df['words'].mean():,.0f}"
)

print(
    f"Median words/document: "
    f"{df['words'].median():,.0f}"
)

print(
    f"Minimum words/document: "
    f"{df['words'].min():,}"
)

print(
    f"Maximum words/document: "
    f"{df['words'].max():,}"
)


# ============================================================
# FILE TYPE DISTRIBUTION
# ============================================================

print("\n===================================")
print("FILE TYPE DISTRIBUTION")
print("===================================")

type_summary = (

    df.groupby("type")
    .agg(
        documents=("source", "count"),
        words=("words", "sum"),
        characters=("characters", "sum")
    )
    .sort_values(
        "words",
        ascending=False
    )

)


type_summary[
    "word_percentage"
] = (

    type_summary["words"]
    /
    total_words
    *
    100

)


print(type_summary)


# ============================================================
# DOCUMENT CONTRIBUTION
# ============================================================

df["word_percentage"] = (

    df["words"]
    /
    total_words
    *
    100

)


df = df.sort_values(
    "words",
    ascending=False
)


print("\n===================================")
print("TOP 20 LARGEST DOCUMENTS")
print("===================================")

for _, row in df.head(20).iterrows():

    print(
        f"{row['source'][:60]:60s} "
        f"{row['words']:10,d} words "
        f"{row['word_percentage']:6.2f}%"
    )


# ============================================================
# TOP DOCUMENT CONCENTRATION
# ============================================================

top_5_words = (
    df.head(5)["words"].sum()
)

top_10_words = (
    df.head(10)["words"].sum()
)

top_20_words = (
    df.head(20)["words"].sum()
)


print("\n===================================")
print("CORPUS CONCENTRATION")
print("===================================")

print(
    f"Top 5 documents: "
    f"{top_5_words / total_words * 100:.2f}%"
)

print(
    f"Top 10 documents: "
    f"{top_10_words / total_words * 100:.2f}%"
)

print(
    f"Top 20 documents: "
    f"{top_20_words / total_words * 100:.2f}%"
)


# ============================================================
# VERY SMALL DOCUMENTS
# ============================================================

small_docs = df[
    df["words"] < 100
]


print("\n===================================")
print("VERY SMALL DOCUMENTS (<100 words)")
print("===================================")

print(
    f"Count: {len(small_docs)}"
)


for _, row in small_docs.iterrows():

    print(
        f"{row['source']:60s} "
        f"{row['words']:6,d} words"
    )


# ============================================================
# SAVE ANALYSIS
# ============================================================

df.to_csv(
    REPORT_OUTPUT,
    index=False,
    encoding="utf-8"
)


print("\n===================================")
print("ANALYSIS COMPLETE")
print("===================================")

print(
    f"Saved to: {REPORT_OUTPUT}"
)