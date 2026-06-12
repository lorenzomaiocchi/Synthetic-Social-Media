import numpy as np
import json 
import pandas as pd
import random 
from agentprofile import create_agents
from ollama import chat


## T = 0 / starting  point, random feed


agents = create_agents(5)

def starting_feed(agents, tweetlist):

    #randomly pick 2 tweets to initialize the setup

    

    startingFeeddict = {}

    for i in agents.keys():

        startingFeeddict[i] = random.sample(tweetlist, 2, )
    return startingFeeddict



fake_tweets = [
    "Sometimes the most productive thing you can do is pause, take a deep breath, and reset. Momentum isn't just about moving fast; it's about moving in the right direction.",
    "Small habits, big results. You don’t need to overhaul your entire life overnight—just focus on winning the next hour. Consistency is the real secret",
    "Reminder: You are allowed to outgrow versions of yourself that no longer serve your peace. Change isn't a loss; it's an evolution",
    "Note to self: Don’t compare your “behind-the-scenes” to everyone else’s highlight reel. You’re exactly where you need to be on your own timeline"
]


feed0 = starting_feed(agents, fake_tweets)

def initial_response(agents, starting_feed):

    responseData = {}


    for a, f in starting_feed.items():


        inputdata = f'''You are: {a} 
            with belief: {agents[a]['belief']} and opinion: {agents[a]['opinion']} 
            
            You read the following tweets: {f}. 
            
            Generate a short response to the tweet
    '''
        
        response = chat(
            model = 'llama3.2', 
            messages= [{'role' : 'user', 'content': inputdata}]
        )

        responseData[a] = response.message.content


    return responseData        



def randomizeFeed(inputPosts):
    
    posts = {}

    for i in agents.keys():

        posts[i] = random.sample(inputPosts, 2 )
    return posts


def CommentCreation(agents, posts):

    responseComments = {}


    for a, f in posts.items():


        inputdata = f'''You are: {a} 
            with belief: {agents[a]['belief']} and opinion: {agents[a]['opinion']} 
            
            You read the following tweets: {f}. 
            
            Generate a short response to the tweet
    '''
        
        response = chat(
            model = 'llama3.2', 
            messages= [{'role' : 'user', 'content': inputdata}]
        )

        responseComments[a] = response.message.content

    return responseComments         





#MAIN LOOP #######
###################

InResp = initial_response(agents, feed0)


history = {}

for t in range(2):

    history[t]  = {}

    if t == 0:
        source = list(InResp.values())
    else:
        source = list(history[t - 1].values())
    
    feed = randomizeFeed(source)
    history[t] = CommentCreation(agents, feed)

    print(f'------ turn {t} ---------')
    print(history[t])


