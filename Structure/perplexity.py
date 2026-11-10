import numpy as np
from ollama import chat


def pplPromptin(inputMessage):

    response = chat(
        model='llama3.2',
        messages=[
            {'role': 'user', 'content': inputMessage}
        ],
        logprobs=True,
        top_logprobs=0,
        options={
            'num_predict': 1,
            'temperature': 1
        },
        stream=False
    )

    return response
    


def compute_perplexity(text, opinion):

    '''Compute perplexity score, re-use the same logic of the original paper'''

    wholeText = text+opinion
    textTokens = text.split()
    wholeTokens = wholeText.split()

    total_log_prob = 0
    token_count = 0

    for i in range(len(textTokens), len(wholeTokens)):

        prompt = ' '.join(wholeTokens[:i])

        response = pplPromptin(prompt)

        logprob = response.logprobs[0].logprob
        
        total_log_prob +=logprob 
        token_count += 1

    avg_neg_log_likelihood = -total_log_prob / token_count
    perplexity =  np.exp(avg_neg_log_likelihood)

    return perplexity


def compute_opinion(text):

    bridge = "Considering the previous statements, "

    pro = "I\'d add the following thought: I think that Nuclear Energy is really beneficial for society"
    con = "I\'d add the following thought: I think that Nuclear Energy is really dangerous for society"

    proPerplexity = compute_perplexity(text + bridge, pro)
    conPerplexity = compute_perplexity(text + bridge, con)

    opinion = conPerplexity / (proPerplexity + conPerplexity)

    return opinion 


## Change the prompting
