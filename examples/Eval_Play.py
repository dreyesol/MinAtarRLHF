################################################################################################################
# Eval_Play.py
# This script runs a trained DQN agent on the MinAtar environment and logs the returns for each episode.
# The script loads the trained model from a checkpoint file and uses it to select actions in the environment.
################################################################################################################
from minatar import Environment
from dqn import *

NUM_EPISODES = 1000

env = Environment("space_invaders")

e = 0
returns = []
num_actions = env.num_actions()

# Get channels and number of actions specific to each game
in_channels = env.state_shape()[2]
num_actions = env.num_actions()
policy_net = QNetwork(in_channels, num_actions).to(device)
# Load model and optimizer if load_path is not None
checkpoint = torch.load("space_invaders_checkpoint")
policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
policy_net.eval()
# Run NUM_EPISODES episodes and log all returns

while e < NUM_EPISODES:
    # Initialize the return for every episode
    G = 0.0
    F = 0
    # Initialize the environment
    env.reset()
    terminated = False

    while(not terminated):
        # Select an action uniformly at random
        # Obtain environment state for policy calculation using QNetwork
        s = get_state(env.state())
        with torch.no_grad():
            action = policy_net(s).max(1)[1].view(1, 1)
            # Act according to the action and observe the transition and reward
            reward, terminated = env.act(action)
            # uncomment to watch the game play
            # env.display_state(80)
            G += reward
        F += 1

    # Increment the episodes
    e += 1
    # Store the return for each episode
    returns.append(G)
    # Uncomment to print the return and steps for each episode
    #print("Episode: " + str(e) + " Return: " + str(G) + " Frames: " + str(F))

average_episode_length = F / NUM_EPISODES
print("Avg Return: " + str(numpy.mean(returns))+"+/-"+str(numpy.std(returns)/numpy.sqrt(NUM_EPISODES)) + " Avg Steps: " + str(average_episode_length))


