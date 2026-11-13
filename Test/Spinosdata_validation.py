import pandas as pd
from Structure.text_processing import load_tokenizer, InferenceModel
from Structure.perplexity import compute_opinion
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
MODEL_LABEL = "qwen"

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "Test" / "Data"

'''Validate the posts using the data from the SPINOS dataset with official labels'''

spinos_official = pd.read_pickle('Test/Data/SPINOSDATA/Spinos_official_dataset_v1.1.pkl')

#Take only content, annotation and topic=='Nuclear'


spinos_official = spinos_official[['content', 'topic', 'annotation']].query("topic=='nuclear-energy' & annotation!='stance_not_inferrable'")


'''There are 101 collected posts on the topic of nuclear energy. We retain just the labelled posts'''

#Check the baseline

#print(spinos_official['annotation'].value_counts())
#print('NORMALIZED SCORES')
#print(spinos_official['annotation'].value_counts(normalize = True))

#Pass the posts to the opinion function

tokenizer = load_tokenizer()
model = InferenceModel()



#this might take a while (not optimal but necessary)
spinos_official['score'] = spinos_official['content'].apply(compute_opinion, args= (tokenizer, model))

output_dir = DATA_DIR / "Results"
output_dir.mkdir(parents=True, exist_ok=True)

output_file = output_dir / f"{MODEL_LABEL}_scores.csv"

spinos_official.to_csv(output_file, index=True)

print(f"Saved {len(spinos_official)} scored posts to {output_file}")