from text_processing import load_tokenizer, InferenceModel
from perplexity import conditional_loss, compute_perplexity


tokenizer = load_tokenizer()
model = InferenceModel()

b = tokenizer('I love cats, they are so cute!', add_special_tokens = False)

print(b)