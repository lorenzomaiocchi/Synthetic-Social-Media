
import json
import random
import numpy as np


#### GENERATE AGENT PROFILE JSON.


mapping_dict = {
    1: 'Strong opposition',
    2: 'Strong opposition',
    3: 'Mild opposition',
    4: 'Mild opposition',
    5: 'Neutral/Mixed',
    6: 'Neutral/Mixed',
    7: 'Mild support',
    8: 'Mild support',
    9: 'Strong support',
    10:'Strong support'
}

def create_agents(n_agents):

    agents = [ f'Agent {i + 1}' for i in range(n_agents)]
    
    

    personality_dict = {}

    for agent in agents:
        mapping_key = int(np.random.choice(list(mapping_dict.keys())))
        
        personality_dict[agent] = {
            'Opennes': np.random.rand(1,1).round(2).tolist(),
            'Ideology':  mapping_key,
            'Positioning': mapping_dict[mapping_key],
            'follow': np.random.choice([a for a in agents if a != agent], 3, replace= False).tolist(),
            'Alpha': 0.10
        }

    print(personality_dict)
    
    return personality_dict

def randomizeFeed(inputPosts):
    
    posts = {}

    for i in agents.keys():

        posts[i] = random.sample(inputPosts, 2 )
    return posts

