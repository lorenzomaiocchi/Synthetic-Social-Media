from text_processing import load_tokenizer, InferenceModel, initial_response
from agentprofile import create_agents
from structure import starting_feed


agents = create_agents(4)
fake_tweets = [
    "Nuclear energy is really dangerous and evil.",
    "They want to talk about nuclear energy but never bring up the solution to nuclear waste #wwastedmoney",
    "Nuclear energy is the only solution possible to climate change",
    "If we had nuclear energy everyone would be better off #changemymind"
]

tokenizer = load_tokenizer()
model = InferenceModel()



feed0 = starting_feed(agents, fake_tweets) #creates the randomized starting feed
InResp = initial_response(agents, feed0, tokenizer, model)