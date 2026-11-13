from text_processing import load_tokenizer, InferenceModel
from perplexity import compute_opinion


tokenizer = load_tokenizer('meta-llama/Llama-3.2-3B-Instruct')
model = InferenceModel('meta-llama/Llama-3.2-3B-Instruct')


posts = {
    "Strong opposition":
        "I completely oppose nuclear energy. "
        "We must shut down all nuclear power plants.",

    "Mild opposition":
        "Nuclear energy presents serious risks. "
        "We should prioritize renewable alternatives.",

    "Neutral":
        "Nuclear energy has both advantages and disadvantages. "
        "Its future requires careful evaluation.",

    "Mild support":
        "Nuclear energy can help reduce emissions, "
        "provided we maintain strict safety standards.",

    "Strong support":
        "I strongly support expanding nuclear energy. "
        "We must build more nuclear power plants "
        "because nuclear power is essential for our future."
}

for label, post in posts.items():
    score = compute_opinion(post, tokenizer, model)
    print(f"{label:20} → {score:.4f}")