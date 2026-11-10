import numpy as np
import json 
import pandas as pd
import random 
from agentprofile import create_agents
from ollama import chat
from perplexity import compute_opinion
import pathlib
from structure import (
    starting_feed,
    initial_response,
    NewFeed,
    CommentCreation,
    opinion,
    opinion_matrix,
    Adjacency_Matrix,
    update_followers,
    FollowIdeology,
    MeanUpdating,
    update_positioning,
    ideology_snap,
    SavefileFunction,
    ComputeInitAdjecency
)



#MAIN LOOP TEST #######
###################
agents = create_agents(4)
fake_tweets = [
    "Nuclear energy is really dangerous and evil.",
    "They want to talk about nuclear energy but never bring up the solution to nuclear waste #wwastedmoney",
    "Nuclear energy is the only solution possible to climate change",
    "If we had nuclear energy everyone would be better off #changemymind"
]


feed0 = starting_feed(agents, fake_tweets) #creates the randomized starting feed
InResp = initial_response(agents, feed0)

history = {}

opinion_history = {
    agent: []
    for agent in agents
}


initial_opinon = opinion(InResp, opinion_history)

ideology_history= {a:[data['Ideology']] for a, data in agents.items()}

N = len(agents.keys())

T = 3

initial_adjecency = ComputeInitAdjecency(agents)

diff_tensor = np.zeros((N, N, T))
adj_tensor = np.zeros((N, N, T), dtype= int)


adj_tensor[: , : , 0] = initial_adjecency



for t in range(1, T + 1):

    history[t]  = {}

    if t == 0:
        source = InResp
    else:
        source = history[t - 1]
    
    feed = NewFeed(agents, source)
    history[t] = CommentCreation(agents, feed)
    Opinion = opinion(history[t], opinion_history=opinion_history)

    Difference_matrix = opinion_matrix(Opinion, t)

    

    AdjMatrix = Adjacency_Matrix(Difference_matrix)
    agents = update_followers(agents, AdjMatrix)
    NewIdeology = FollowIdeology(agents)
    agents = MeanUpdating(agents)
    agents = update_positioning(agents)


    #Save the results

    ideology_data = ideology_snap(agents, ideology_history)
    adj_tensor[: , :, t] = AdjMatrix
    diff_tensor[: , :, t] = Difference_matrix





    print(f'------ turn {t} ---------')
    print(opinion_history)

#SavefileFunction(ideology_data, 'Ideology_history.json')
#SavefileFunction(history, 'Comments_history.json')
#SavefileFunction(opinion_history, 'Opinion_history.json')
#np.savez_compressed('Analysis/Data/network_history.npz', adj_tensor = adj_tensor, diff_tensor = diff_tensor)
