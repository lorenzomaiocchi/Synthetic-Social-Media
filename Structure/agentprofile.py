
import json
import random
import numpy as np


#### GENERATE AGENT PROFILE JSON.


def create_agents(n_agents):

    agents = [ f'Agent {i + 1}' for i in range(n_agents)]

    personality_dict = {}

    for agent in agents:
        personality_dict[agent] = {
            'belief': np.random.rand(1,1).round(2).tolist(),
            'opinion':  np.random.rand(1, 1).round(2).tolist()
        }

    
    
    return personality_dict




create_agents(10)
