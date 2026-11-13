import argparse
import pandas as pd
from pathlib import Path

from Structure.text_processing import load_tokenizer, InferenceModel
from Structure.perplexity import compute_opinion


# Available models
MODELS = {
    "qwen": "Qwen/Qwen2.5-3B-Instruct",
    "llama": "meta-llama/Llama-3.2-3B-Instruct"
}

# Command-line arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    choices=MODELS.keys(),
    default="qwen"
)

args = parser.parse_args()

MODEL_LABEL = args.model
MODEL_NAME = MODELS[MODEL_LABEL]


# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Test" / "Data"
OUTPUT_DIR = PROJECT_ROOT / "Analysis" / "Data"


# Load SPINOS dataset
spinos_official = pd.read_pickle(
    DATA_DIR / "SPINOSDATA" / "SPINOS_official_dataset_v1.1.pkl"
)

# Retain labelled nuclear-energy posts
spinos_official = (
    spinos_official[["content", "topic", "annotation"]]
    .query(
        "topic == 'nuclear-energy' "
        "& annotation != 'stance_not_inferrable'"
    )
    .copy()
)


# Load selected model
print(f"Loading model: {MODEL_NAME}")

tokenizer = load_tokenizer(MODEL_NAME)
model = InferenceModel(MODEL_NAME)


# Compute opinion scores
spinos_official["score"] = spinos_official["content"].apply(
    compute_opinion,
    args=(tokenizer, model)
)


# Save results
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_file = OUTPUT_DIR / f"spinos_{MODEL_LABEL}_scores.csv"

spinos_official.to_csv(output_file, index=False)

print(f"Saved {len(spinos_official)} scored posts to {output_file}")