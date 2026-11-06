import numpy as np
import json 
import pandas as pd
import random 
from agentprofile import create_agents
from ollama import chat
from perplexity import compute_opinion

## T = 0 / starting  point, random feed


agents = create_agents(4)

def starting_feed(agents, tweetlist):

    '''randomly pick 2 tweets for each agent to initialize the setup'''

    

    startingFeeddict = {}

    for i in agents.keys():

        startingFeeddict[i] = random.sample(tweetlist, 2)
    return startingFeeddict



fake_tweets = [
    "Nuclear energy is really dangerous and evil.",
    "They want to talk about nuclear energy but never bring up the solution to nuclear waste #wwastedmoney",
    "Nuclear energy is the only solution possible to climate change",
    "If we had nuclear energy everyone would be better off #changemymind"
]


feed0 = starting_feed(agents, fake_tweets) #creates the randomized starting feed



def initial_response(agents, starting_feed):

    responseData = {}


    for a, f in starting_feed.items():


        inputdata = f'''You are: {a} 
            
            You read the following tweets: {f}. 
            
            Your position on the topic is {agents[a]['Positioning']}

            Generate a short response to the tweet in line with your position on the topic
    '''
        
        response = chat(
            model = 'llama3.2', 
            messages= [{'role' : 'user', 'content': inputdata}],
            options={'temperature': 1}
        )

        responseData[a] = response.message.content


    return responseData        




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

        Outfeed[a] = feed

    return Outfeed


def CommentCreation(agents, posts):

    responseComments = {}


    for a, f in posts.items():


        inputdata = f'''You are: {a} 
            
            You read the following tweets: {f}. 
            
            Your position on the topic is {agents[a]['Positioning']}

            Generate a short response to the tweet in line with your position on the topic
    '''
        
        response = chat(
            model = 'llama3.2', 
            messages= [{'role' : 'user', 'content': inputdata}]
        )

        responseComments[a] = response.message.content

    return responseComments         


def opinion(turn, opinion_history):

    for agent, tweet in turn.items():

        opinion = compute_opinion(tweet)

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






#MAIN LOOP #######
###################

InResp = initial_response(agents, feed0)


history = {}


opinion_history = {
    agent: []
    for agent in agents
}


for t in range(4):

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



    print(f'------ turn {t} ---------')
    #print(history[t])
    print(agents)
