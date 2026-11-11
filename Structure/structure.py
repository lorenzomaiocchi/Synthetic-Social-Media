import numpy as np
import json 
import pandas as pd
import random 
from agentprofile import create_agents
from perplexity import compute_opinion
import pathlib
## T = 0 / starting  point, random feed




def starting_feed(agents, tweetlist):

    '''randomly pick 2 tweets for each agent to initialize the setup'''

    

    startingFeeddict = {}

    for i in agents.keys():

        startingFeeddict[i] = random.sample(tweetlist, 2)
    return startingFeeddict



#def randomizeFeed(inputPosts):
#    
#    posts = {}
#
#    for i in agents.keys():
#
#        posts[i] = random.sample(inputPosts, 2 )
#    return posts


def NewFeed(agents, inputposts):

    '''read n-th agents followers posts'''

    Outfeed = {}

    for a in agents.keys():

        feed = '\n'.join([f'{p} wrote: {inputposts[p]}' for p in agents[a]['follow']])


        Outfeed[a] = {
            'Feed': feed, 
            'Previous_response': inputposts[a]
        }



    return Outfeed



def opinion(turn, opinion_history, tokenizer, model):

    for agent, tweet in turn.items():

        opinion = compute_opinion(tweet, tokenizer, model)

        opinion_history[agent].append(opinion)

    return opinion_history



def opinion_matrix(opinions, t):

    #opinion difference at each time step t
    opinion_array = np.array(list(opinions.values()))

    scores = opinion_array[:, t].flatten()

    return np.abs(
        scores[:, None] - scores[None, :]
    )


def Adjacency_Matrix(differenceMatrix):

    connections = np.where(differenceMatrix < 0.1, 1 , 0 )

    np.fill_diagonal(connections, 0)


    return connections
            


def update_followers(agents, adjacency):

    agent_names = list(agents.keys())

    for i, agent in enumerate(agent_names):

        agents[agent]['follow'] = [
            agent_names[j]
            for j in range(len(agent_names))
            if adjacency[i, j] == 1
        ]

    return agents




def FollowIdeology(agents):

    followIdeology = {}

    for a in agents.keys():
        followers = agents[a]['follow']

        if len(followers) > 0: 
            ideology = [agents[Id]['Ideology'] for Id in followers]

            followIdeology[a] = np.mean(ideology)

        else:
            followIdeology[a] = agents[a]['Ideology']

    return followIdeology


def MeanUpdating(agents):

    idarray = FollowIdeology(agents)
    new_ideologies = {}

    for a in agents.keys():

        u = float(agents[a]['Opennes'][0])
        Id_agent = agents[a]['Ideology']
        
        fl_mean = idarray[a]

        new = ((1 - u) * Id_agent) + (u * fl_mean)

        new_ideologies[a] = np.clip(new, 1,10)

    for a, new in new_ideologies.items():
        agents[a]['Ideology'] = float(new)

    return agents

def update_positioning(agents):

    for a in agents.keys():

        ideology = agents[a]['Ideology']

        if ideology < 2.5:
            positioning = 'Strong opposition'

        elif ideology < 4.5:
            positioning = 'Mild opposition'

        elif ideology < 6.5:
            positioning = 'Neutral/Mixed'

        elif ideology < 8.5:
            positioning = 'Mild support'

        else:
            positioning = 'Strong support'

        agents[a]['Positioning'] = positioning

    return agents



def ideology_snap(agents, ideology_dictionary):

    for a, data in agents.items():

        ideology_dictionary[a].append(data['Ideology'])

    return ideology_dictionary




def SavefileFunction(data, filename, path= 'Analysis/Data'):

    file_path = pathlib.Path(path)

    file_path.mkdir(parents= True, exist_ok=True)

    savefile = file_path / filename

    with open(savefile, 'w', encoding='utf-8') as f:
        json.dump(data, f )



def ComputeInitAdjecency(agents) -> np.array:

    agent_names = list(agents.keys())

    initial_adjacency = np.array([
    [
        int(other in agents[agent]['follow'])
        for other in agent_names
    ]
    for agent in agent_names
    ], dtype=int)

    return initial_adjacency

