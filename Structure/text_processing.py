'''Functions to generate text and culculating perplexity using transformer library

"Qwen/Qwen2.5-3B-Instruct" is the default model

'''


import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_tokenizer(altmodel=None):
    return AutoTokenizer.from_pretrained(altmodel or "Qwen/Qwen2.5-3B-Instruct")


def InferenceModel(altmodel=None, dtype="auto"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = AutoModelForCausalLM.from_pretrained(
        altmodel or "Qwen/Qwen2.5-3B-Instruct",
        dtype=dtype,
        device_map="auto"
    )

    model.eval()
    return model


def initial_response(agents,starting_feed):

    responseData = {}


    for a, f in starting_feed.items():


        message = f'''
        You are a user participating in a social media discussion about nuclear energy.
            
        Your position on the topic is {agents[a]['Positioning']}
            
        You read the following tweets: 
            {f}. 
            

        Write ONE new social media post about nuclear energy.
            
                    Rules:
                    - Express an opinion consistent with your assigned position.
                    - You may agree or disagree with the posts you read.
                    - Write no more than 40 words.
                    - Write only the post itself.
                    - Do not explain your instructions or describe what you are doing.
                    - Do not generate fictional conversations or quote imaginary replies.
                    - Do not include labels such as "Tweet:" or "Response:". '''


        inputdata = [{'role': 'user', 'content': (message)}]

        data = tokenizer.apply_chat_template(
            inputdata,
            add_generation_prompt = True,
            tokenize = True,
            return_dict = True,
            return_tensor = 'pt').to(model.device)

        with torch.inference_mode():
            outputs = model.generate(
            **data,
            max_new_tokens=70,
            do_sample=True,
            temperature=0.8,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id
    )

    print(outputs)




    return responseData        



