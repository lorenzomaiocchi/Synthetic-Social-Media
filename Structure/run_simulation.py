
import argparse
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
# DEFAULT CONFIGURATION
# ============================================================

FAKE_TWEETS = [
    "Nuclear energy is really dangerous and evil.",
    "They want to talk about nuclear energy but never bring up the solution to nuclear waste #wwastedmoney",
    "Nuclear energy is the only solution possible to climate change",
    "If we had nuclear energy everyone would be better off #changemymind",
]


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Run the Qwen-based synthetic social media simulation."
    )

    parser.add_argument(
        "--agents",
        type=int,
        default=4,
        help="Number of agents (default: 4).",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=3,
        help="Number of simulation steps (default: 3).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="Analysis/Data",
        help="Directory for simulation results.",
    )

    parser.add_argument(
        '--model',
        type = str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help= 'Hugging-face model identifier'
    )

    args = parser.parse_args()

    if args.agents < 3:
        parser.error("--agents must be at least 3.")

    if args.steps < 1:
        parser.error("--steps must be at least 1.")

    return args


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# SIMULATION
# ============================================================

def run_simulation(n_agents, steps, seed, output_dir, model_name="Qwen/Qwen2.5-3B-Instruct"):

    set_seed(seed)

    print("=" * 50)
    print("SYNTHETIC SOCIAL MEDIA SIMULATION")
    print("=" * 50)
    print(f"Agents:     {n_agents}")
    print(f"Steps:      {steps}")
    print(f"Seed:       {seed}")
    print(f"Output:     {output_dir}")
    print("=" * 50)

    # --------------------------------------------------------
    # 1. Load model and tokenizer
    # --------------------------------------------------------

    print(f"\nLoading model and tokenizer: {model_name}")

    tokenizer = load_tokenizer()
    model = InferenceModel()

    # --------------------------------------------------------
    # 2. Create agents
    # --------------------------------------------------------

    agents = create_agents(n_agents)

    agent_names = list(agents.keys())
    N = len(agent_names)

    # --------------------------------------------------------
    # 3. Generate initial posts
    # --------------------------------------------------------

    print("\nGenerating initial posts...")

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

    # Score initial posts
    opinion(
        initial_posts,
        opinion_history,
        tokenizer,
        model,
    )

    # --------------------------------------------------------
    # 4. Initialize network history
    # --------------------------------------------------------

    diff_tensor = np.zeros(
        (N, N, steps + 1),
        dtype=float,
    )

    adj_tensor = np.zeros(
        (N, N, steps + 1),
        dtype=int,
    )

    adj_tensor[:, :, 0] = ComputeInitAdjecency(agents)

    diff_tensor[:, :, 0] = opinion_matrix(
        opinion_history,
        0,
    )

    # --------------------------------------------------------
    # 5. Main simulation loop
    # --------------------------------------------------------

    for t in range(1, steps + 1):

        print(f"\n{'=' * 50}")
        print(f"TIME STEP {t}/{steps}")
        print(f"{'=' * 50}", flush=True)

        # Read previous posts
        previous_posts = history[t - 1]

        # Build feeds from followed accounts
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

        # Calculate pairwise opinion differences
        difference_matrix = opinion_matrix(
            opinion_history,
            t,
        )

        # Rewire social network
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

        # Record updated ideologies
        ideology_snap(
            agents,
            ideology_history,
        )

        # Record network state
        adj_tensor[:, :, t] = adjacency_matrix
        diff_tensor[:, :, t] = difference_matrix

        # ----------------------------------------------------
        # Progress summary
        # ----------------------------------------------------

        current_opinions = {
            agent: scores[-1]
            for agent, scores in opinion_history.items()
        }

        current_ideologies = {
            agent: data["Ideology"]
            for agent, data in agents.items()
        }

        n_connections = int(adjacency_matrix.sum())

        print("\nPosts:")
        for agent, post in current_posts.items():
            print(f"{agent}: {post}")

        print("\nOpinions:")
        print(current_opinions)

        print("\nIdeologies:")
        print(current_ideologies)

    # --------------------------------------------------------
    # 6. Save results
    # --------------------------------------------------------

    print("\nSaving simulation results...")

    SavefileFunction(
        ideology_history,
        "Ideology_history.json",
        path=output_dir,
    )

    SavefileFunction(
        history,
        "Comments_history.json",
        path=output_dir,
    )

    SavefileFunction(
        opinion_history,
        "Opinion_history.json",
        path=output_dir,
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        output_path / "network_history.npz",
        adj_tensor=adj_tensor,
        diff_tensor=diff_tensor,
    )

    # Save experiment configuration
    SavefileFunction(
        {
            "n_agents": n_agents,
            "steps": steps,
            "seed": seed,
            "model": "Qwen/Qwen2.5-3B-Instruct",
        },
        "config.json",
        path=output_dir,
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

    args = parse_args()

results = run_simulation(
    n_agents=args.agents,
    steps=args.steps,
    seed=args.seed,
    output_dir=args.output_dir,
    model_name=args.model,
)
