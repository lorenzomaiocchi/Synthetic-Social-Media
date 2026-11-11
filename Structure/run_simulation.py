
import random
from pathlib import Path

import numpy as np
import torch

from agentprofile import create_agents

from text_processing import (
    load_tokenizer,
    InferenceModel,
    initial_response,
    CommentCreation,
)

from structure import (
    starting_feed,
    NewFeed,
    opinion,
    opinion_matrix,
    Adjacency_Matrix,
    update_followers,
    MeanUpdating,
    update_positioning,
    ideology_snap,
    SavefileFunction,
    ComputeInitAdjecency,
)


# ============================================================
# CONFIGURATION
# ============================================================

N_AGENTS = 4
T = 3
SEED = 42

OUTPUT_DIR = "Analysis/Data"

FAKE_TWEETS = [
    "Nuclear energy is really dangerous and evil.",
    "They want to talk about nuclear energy but never bring up the solution to nuclear waste #wwastedmoney",
    "Nuclear energy is the only solution possible to climate change",
    "If we had nuclear energy everyone would be better off #changemymind",
]


# ============================================================
# SIMULATION
# ============================================================

def run_simulation():

    # --------------------------------------------------------
    # 1. Reproducibility
    # --------------------------------------------------------

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    # --------------------------------------------------------
    # 2. Load model and tokenizer once
    # --------------------------------------------------------

    print("Loading Qwen model and tokenizer...")

    tokenizer = load_tokenizer()
    model = InferenceModel()

    # --------------------------------------------------------
    # 3. Create agents
    # --------------------------------------------------------

    agents = create_agents(N_AGENTS)

    agent_names = list(agents.keys())
    N = len(agent_names)

    # --------------------------------------------------------
    # 4. Initialize simulation
    # --------------------------------------------------------

    feed0 = starting_feed(agents, FAKE_TWEETS)

    initial_posts = initial_response(
        agents,
        feed0,
        tokenizer,
        model,
    )

    history = {0: initial_posts}

    opinion_history = {
        agent: []
        for agent in agent_names
    }

    ideology_history = {
        agent: [data["Ideology"]]
        for agent, data in agents.items()
    }

    # Measure opinions at t = 0
    opinion(
        initial_posts,
        opinion_history,
        tokenizer,
        model,
    )

    # --------------------------------------------------------
    # 5. Initialize network history
    # --------------------------------------------------------

    diff_tensor = np.zeros(
        (N, N, T + 1),
        dtype=float,
    )

    adj_tensor = np.zeros(
        (N, N, T + 1),
        dtype=int,
    )

    adj_tensor[:, :, 0] = ComputeInitAdjecency(agents)

    # Initial opinion differences
    diff_tensor[:, :, 0] = opinion_matrix(
        opinion_history,
        0,
    )

    # --------------------------------------------------------
    # 6. Main simulation loop
    # --------------------------------------------------------

    for t in range(1, T + 1):

        print(f"\n{'=' * 50}")
        print(f"TIME STEP {t}/{T}")
        print(f"{'=' * 50}")

        # Previous time step's posts
        previous_posts = history[t - 1]

        # Construct social feeds
        feed = NewFeed(agents, previous_posts)

        # Generate new posts
        current_posts = CommentCreation(
            agents,
            feed,
            tokenizer,
            model,
        )

        history[t] = current_posts

        # Measure expressed opinions
        opinion(
            current_posts,
            opinion_history,
            tokenizer,
            model,
        )

        # Calculate opinion distances
        difference_matrix = opinion_matrix(
            opinion_history,
            t,
        )

        # Update network
        adjacency_matrix = Adjacency_Matrix(
            difference_matrix
        )

        agents = update_followers(
            agents,
            adjacency_matrix,
        )

        # Update latent ideologies
        agents = MeanUpdating(agents)
        agents = update_positioning(agents)

        # Record history
        ideology_snap(
            agents,
            ideology_history,
        )

        adj_tensor[:, :, t] = adjacency_matrix
        diff_tensor[:, :, t] = difference_matrix

        # ----------------------------------------------------
        # Debug output
        # ----------------------------------------------------

        current_opinions = {
            agent: scores[-1]
            for agent, scores in opinion_history.items()
        }

        current_ideologies = {
            agent: data["Ideology"]
            for agent, data in agents.items()
        }

        print("\nPosts:")
        for agent, post in current_posts.items():
            print(f"{agent}: {post}")

        print("\nOpinions:")
        print(current_opinions)

        print("\nAdjacency matrix:")
        print(adjacency_matrix)

        print("\nIdeologies:")
        print(current_ideologies)

    # --------------------------------------------------------
    # 7. Save results
    # --------------------------------------------------------

    print("\nSaving simulation results...")

    SavefileFunction(
        ideology_history,
        "Ideology_history.json",
        path=OUTPUT_DIR,
    )

    SavefileFunction(
        history,
        "Comments_history.json",
        path=OUTPUT_DIR,
    )

    SavefileFunction(
        opinion_history,
        "Opinion_history.json",
        path=OUTPUT_DIR,
    )

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path / "network_history.npz",
        adj_tensor=adj_tensor,
        diff_tensor=diff_tensor,
    )

    print("\nSimulation completed successfully!")

    return {
        "agents": agents,
        "posts": history,
        "opinions": opinion_history,
        "ideologies": ideology_history,
        "adjacency": adj_tensor,
        "differences": diff_tensor,
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    results = run_simulation()
