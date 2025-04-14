################################################################################################################
# python3 dqn.py -g <game>                                                                                     #
#   -o, --output <directory/file name prefix>                                                                  #
#   -v, --verbose: outputs the average returns every 1000 episodes                                             #
#   -l, --loadfile <directory/file name of the saved model>                                                    #
#   -a, --alpha <number>: step-size parameter                                                                  #
#   -s, --save: save model data every 1000 episodes                                                            #
#   -r, --replayoff: disable the replay buffer and train on each state transition                              #
#   -t, --targetoff: disable the target network                                                                #
#                                                                                                              #
# References used for this implementation:                                                                     #
#   dqn.py by Kenny Young (kjyoung@ualberta.ca) and Tian Tian(ttian@ualberta.ca)                               #
#   https://pytorch.org/docs/stable/nn.html#                                                                   #
#   https://pytorch.org/docs/stable/torch.html                                                                 #
#   https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html                                   #
################################################################################################################

import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import time

import copy

import random, numpy, argparse, logging, os

from collections import namedtuple
from minatar import Environment

################################################################################################################
# Constants
#
################################################################################################################
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 100000
TARGET_NETWORK_UPDATE_FREQ = 1000
TRAINING_FREQ = 1
NUM_FRAMES = 50000000
FIRST_N_FRAMES = 100000
REPLAY_START_SIZE = 5000
END_EPSILON = 0.1
STEP_SIZE = 0.0000003
SQUARED_GRAD_MOMENTUM = 0.95
MIN_SQUARED_GRAD = 0.01
GAMMA = 0.99
EPSILON = 0.1
FREEZE_CONV = False
LOAD_BASELINE = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


################################################################################################################
# class QNetworkPbTS
#
# A class that implements the bootstrapped QNetwork architecture, which is a convolutional neural network with one
# convolutional layer and one fully connected layer.  The output of the QNetwork is a list of linear layers, one for
# each action.  The QNetwork is used to approximate the Reward Function.
#
################################################################################################################
class QNetworkPbTS(nn.Module):
    def __init__(self, in_channels, num_actions):

        super(QNetworkPbTS, self).__init__()

        # One hidden 2D convolution layer:
        #   in_channels: variable
        #   out_channels: 16
        #   kernel_size: 3 of a 3x3 filter matrix
        #   stride: 1
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, stride=1)

        # Final fully connected hidden layer:
        #   the number of linear unit depends on the output of the conv
        #   the output consist 128 rectified units
        def size_linear_unit(size, kernel_size=3, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1
        num_linear_units = size_linear_unit(10) * size_linear_unit(10) * 16
        self.fc_hidden = nn.Linear(in_features=num_linear_units, out_features=128)

        # Output layer:class MyModule(nn.Module):
        self.linearHeads = nn.ModuleList([nn.Linear(in_features=128, out_features=num_actions) for i in range(8)])

    def forward(self, x, head):
        # Rectified output from the first conv layer
        x = f.relu(self.conv(x))

        # Rectified output from the final hidden layer
        x = f.relu(self.fc_hidden(x.view(x.size(0), -1)))

        # Returns the output from the fully-connected linear layer
        return self.linearHeads[head](x)

###########################################################################################################
# class replay_buffer
#
# A cyclic buffer of a fixed size containing the last N number of recent transitions.  A transition is a
# tuple of state, next_state, action, reward, is_terminal.  The boolean is_terminal is used to indicate
# whether if the next state is a terminal state or not.
#
###########################################################################################################
transition = namedtuple('transition', 'state, next_state, action, reward, is_terminal')
class replay_buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.location = 0
        self.buffer = []

    def add(self, *args):
        # Append when the buffer is not full but overwrite when the buffer is full
        if len(self.buffer) < self.buffer_size:
            self.buffer.append(transition(*args))
        else:
            self.buffer[self.location] = transition(*args)

        # Increment the buffer location
        self.location = (self.location + 1) % self.buffer_size

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)


################################################################################################################
# get_state
#
# Converts the state given by the environment to a tensor of size (in_channel, 10, 10), and then
# unsqueeze to expand along the 0th dimension so the function returns a tensor of size (1, in_channel, 10, 10).
#
# Input:
#   s: current state as numpy array
#
# Output: current state as tensor, permuted to match expected dimensions
#
################################################################################################################
def get_state(s):
    return (torch.tensor(s, device=device).permute(2, 0, 1)).unsqueeze(0).float()


################################################################################################################
# world_dynamics
#
# It generates the next state and reward after taking an action according to the behavior policy.  The behavior
# policy is epsilon greedy: epsilon probability of selecting a random action and 1 - epsilon probability of
# selecting the action with max Q-value.
#
# Inputs:
#  env: environment of the game
#  policy_net: policy network, an instance of QNetworkPbTS
#  PolicyHead: head of the policy network to use
#  ComparatorHead: head of the comparator network to use
#
# Output: PolicyReward, ComparatorReward, trueReward, is_terminated
#
################################################################################################################
def world_dynamics(env, policy_net, PolicyHead, ComparatorHead):

    s = get_state(env.state())

    with torch.no_grad():
        policyPredictedRewards = policy_net(s, PolicyHead)
        ComparatorHeadReward = policy_net(s, ComparatorHead)
        action = policyPredictedRewards.max(1)[1].view(1, 1)
        PolicyReward = policyPredictedRewards[0][action.item()].view(1, 1)
        ComparatorReward = ComparatorHeadReward[0][action.item()].view(1, 1)

    # Act according to the action and observe the transition and reward
    trueReward, terminated = env.act(action)

    return PolicyReward, ComparatorReward, trueReward, terminated


################################################################################################################
# train
#
# This is where learning happens. More specifically, this function learns the weights of the policy network
# using huber loss.
#
# Inputs:
#   state_buffer: a list of states to sample from
#   comparator_return: the return of the comparator network
#   policy_net: the policy network, an instance of QNetworkPbTS
#   policy_head: the head of the policy network to use
#   preference: the preference obtained from feedback
#   optimizer: the optimizer to use for training the policy network
#
################################################################################################################
def train(state_buffer, comparator_return, policy_net, policy_head, preference, optimizer):
    policy_return = policy_net(state_buffer, policy_head).max(1)[0].sum(0)
    # Compute loss
    criterion = nn.MarginRankingLoss(margin=0.1)
    loss = criterion(comparator_return, policy_return, preference)

    # Zero gradients, backprop, update the weights of policy_net
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


################################################################################################################
# dqn
#
# DQN algorithm with the option to disable replay and/or target network, and the function saves the training data.
#
# Inputs:
#   output_file_name: directory and file name prefix to output data and network weights, file saved as 
#       <output_file_name>_data_and_weights
#   store_intermediate_result: a boolean, if set to true will store checkpoint data every 1000 episodes
#       to a file named <output_file_name>_checkpoint
#   load_path: file path for a checkpoint to load, and continue training from
#
#################################################################################################################
def dqn(output_file_name, store_intermediate_result=False, load_path=None):
    epsilon = EPSILON
    env0 = Environment("space_invaders")
    # uncomment the following line to watch the game being played while training
    # testEnv = Environment("space_invaders")

    # Get channels and number of actions specific to each game
    in_channels = env0.state_shape()[2]
    num_actions = env0.num_actions()

    # Instantiate networks, optimizer, loss and buffer
    policy_net = QNetworkPbTS(in_channels, num_actions).to(device)
    
    #sample initial arbitrary state to initialize the network heads from a uniform distribution with discrete values from 0 to 7
    head0 = torch.randint(0, 7, (1,1), device=device).item()
    head1 = torch.randint(0, 7, (1,1), device=device).item()

    optimizer = optim.RMSprop(policy_net.parameters(), lr=STEP_SIZE, alpha=SQUARED_GRAD_MOMENTUM, centered=True, eps=MIN_SQUARED_GRAD)

    # Set initial values
    e_init = 0
    t_init = 0
    policy_net_update_counter_init = 0
    avg_return_init = 0.0
    data_return_init = []
    frame_stamp_init = []

    # Load model and optimizer if load_path is not None
    if load_path is not None and isinstance(load_path, str):
        checkpoint = torch.load(load_path)
        if LOAD_BASELINE:
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'], strict=False)
            for head in policy_net.linearHeads:
                head.weight.data = checkpoint['policy_net_state_dict']['output.weight'].to(device)
                head.bias.data = checkpoint['policy_net_state_dict']['output.bias'].to(device)
        else:
            policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            e_init = checkpoint['episode']
            t_init = checkpoint['frame']
            policy_net_update_counter_init = checkpoint['policy_net_update_counter']
            avg_return_init = checkpoint['avg_return']
            data_return_init = checkpoint['return_per_run']
            frame_stamp_init = checkpoint['frame_stamp_per_run']

    # Set to training mode
    policy_net.train()
    if FREEZE_CONV:
        for param in policy_net.conv.parameters():
            param.requires_grad = False
    # Data containers for performance measure and model related data
    data_return = data_return_init
    frame_stamp = frame_stamp_init
    avg_return = avg_return_init
    avgFramesSurvived = 0

    # Train for a number of frames
    t = t_init
    e = e_init
    policy_net_update_counter = policy_net_update_counter_init
    t_start = time.time()
    testTerminated = False
    while t < NUM_FRAMES:
        # Initialize the return for every trajectory (we should see this eventually increase)
        trajectory0ReturnOnPolicyHead = 0.0
        trajectory0ReturnOnComparatorHead = 0.0
        trajectory1ReturnOnPolicyHead = 0.0
        trajectory1ReturnOnComparatorHead = 0.0
        trajectory0TrueReward = 0
        trajectory1TrueReward = 0
        G = 0
        # Initialize the environment and start state
        env0.reset()
        env1 = copy.deepcopy(env0)
        is_terminated = False
        is_terminated0 = False
        is_terminated1 = False
        train_s0_buffer = None
        framesSurvived = 0
        # uncomment the following lines to watch the game being played while training
        # if e % 200 == 1:
        #     testTerminated = False
        #     testEnv.reset()
        #     totalReward = 0
        #     while not testTerminated:
        #         s = get_state(testEnv.state())
        #         with torch.no_grad():
        #             action = policy_net(s, head0).max(1)[1].view(1, 1)
        #             # Act according to the action and observe the transition and reward
        #         reward, testTerminated = testEnv.act(action)
        #         totalReward += reward
        #         testEnv.display_state(20)
                
        #     testEnv.close_display()
        #     print("Test Return: " + str(totalReward) + " Head Used: " + str(head0))

        while(not is_terminated) and t < NUM_FRAMES:
            # create trajectories by selecting the best action according to the policy network
            # and then take the action in each environment
            if not is_terminated0:
                rewardOnPolicyHead, rewardOnComparatorHead, trueReward, is_terminated0 = world_dynamics(env0, policy_net, head0, head1)
                # if the buffer is empty, initialize it with the first state
                if train_s0_buffer is None:
                    train_s0_buffer = get_state(env0.state())
                else:
                    train_s0_buffer = torch.cat((train_s0_buffer, get_state(env0.state())), dim=0)
                trajectory0ReturnOnComparatorHead += rewardOnComparatorHead.item()
                trajectory0ReturnOnPolicyHead += rewardOnPolicyHead.item()
                trajectory0TrueReward += trueReward

            if not is_terminated1:
                rewardOnComparatorHead, rewardOnPolicyHead, trueReward, is_terminated1 = world_dynamics(env1, policy_net, head1, head0)
                trajectory1ReturnOnComparatorHead += rewardOnComparatorHead.item()
                trajectory1ReturnOnPolicyHead += rewardOnPolicyHead.item()
                trajectory1TrueReward += trueReward
            
            #if both environments are terminated, then we can stop the episode
            # and query for preference feedback
            is_terminated = is_terminated0 and is_terminated1
            # calculate uncertainity
            uncertainity = abs(trajectory0ReturnOnPolicyHead - trajectory1ReturnOnPolicyHead - (trajectory0ReturnOnComparatorHead - trajectory1ReturnOnComparatorHead))

            if uncertainity > epsilon or is_terminated:
                #query for preference feedback
                preference = queryForPreference(trajectory0TrueReward, trajectory1TrueReward, is_terminated0, is_terminated1)
                if preference == -1:
                    G += trajectory0ReturnOnPolicyHead
                else:
                    G += trajectory1ReturnOnComparatorHead
                # make sure trajectory return and preference are tensors
                trajectory1ReturnOnComparatorHead = torch.tensor(trajectory1ReturnOnComparatorHead, device=device).float()
                preference = torch.tensor(preference, device=device).float()
                # train the network
                train(train_s0_buffer, trajectory1ReturnOnComparatorHead, policy_net, head0, preference, optimizer)

                # after training, restart the trajectories and copy train environment to comparator environment
                trajectory0ReturnOnPolicyHead = 0.0
                trajectory0ReturnOnComparatorHead = 0.0
                trajectory1ReturnOnPolicyHead = 0.0
                trajectory1ReturnOnComparatorHead = 0.0
                trajectory0TrueReward = 0
                trajectory1TrueReward = 0
                train_s0_buffer = None
                # if train environment is not terminated, then copy the state to comparator environment
                # otherwise, set the episode to terminated
                if not is_terminated0:
                    env1 = copy.deepcopy(env0)
                else:
                    is_terminated = True
                # after training, determine the heads to use for next trajectories
                head1 = head0
                initial_state = get_state(env0.state())
                #pick new head for the policy network
                with torch.no_grad():
                    BestReward = None
                    for head, _ in enumerate(policy_net.linearHeads):
                        sampledHeadReward = policy_net(initial_state, head)[0][numpy.random.randint(num_actions)]
                        if BestReward is not None:
                            if sampledHeadReward > BestReward:
                                BestReward = sampledHeadReward
                                head0 = head
                        else:
                            BestReward = sampledHeadReward
                            head0 = head
            #increment step at the end of each frame
            t += 1
            framesSurvived += 1
        # Increment the episodes
        e += 1

        # Save the return for each episode
        data_return.append(G)
        frame_stamp.append(t)

        # Logging exponentiated return only when verbose is turned on and only at 1000 episode intervals
        avg_return = 0.99 * avg_return + 0.01 * G
        avgFramesSurvived = 0.99 * avgFramesSurvived + 0.01 * framesSurvived
        if e % 20 == 0:
            logging.info("Episode " + str(e) + " | Avg Frames Survived Per Episode: " +
                         str(numpy.around(avgFramesSurvived, 2)) + " | Avg Return: " + str(numpy.around(avg_return,2)))

        # Save model data and other intermediate data if the corresponding flag is true
        if store_intermediate_result and e % 500 == 0:
            torch.save({
                        'episode': e,
                        'frame': t,
                        'policy_net_update_counter': policy_net_update_counter,
                        'policy_net_state_dict': policy_net.state_dict(),
                        'target_net_state_dict': [],
                        'optimizer_state_dict': optimizer.state_dict(),
                        'avg_return': avg_return,
                        'return_per_run': data_return,
                        'frame_stamp_per_run': frame_stamp,
                        'replay_buffer': []
            }, output_file_name + "_" + str(e) + "_checkpoint")

    # Print final logging info
    logging.info("Avg return: " + str(numpy.around(avg_return, 2)) + " | Time per frame: " + str((time.time()-t_start)/t))
        
    # Write data to file
    torch.save({
        'returns': data_return,
        'frame_stamps': frame_stamp,
        'policy_net_state_dict': policy_net.state_dict()
    }, output_file_name + "_data_and_weights")
#######################################################################################################################
# queryForPreference
#
# This function queries for preference feedback from the user.  The function should be replaced with a human feedback
# mechanism.  For now, it selects the trajectory that did not lose.  If neither lost, the one with the higher true reward.
# The function returns -1 if the first trajectory is preferred and 1 if the second trajectory is preferred.
#
# Inputs:
#   trajectory0TrueReward: the true reward from the environment of the main head trajectory
#   trajectory1TrueReward: the true reward from the environment of the comparator head trajectory
#   is_terminated0: boolean indicating if the first trajectory ends in game over
#   is_terminated1: boolean indicating if the second trajectory ends in game over
#######################################################################################################################
def queryForPreference(trajectory0TrueReward, trajectory1TrueReward, is_terminated0, is_terminated1):
    preference = 1
    if not is_terminated0 and is_terminated1:
        preference = -1
    elif is_terminated0 and is_terminated1:
        if trajectory0TrueReward > trajectory1TrueReward:
            preference = -1

    return preference

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", "-o", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--loadfile", "-l", type=str)
    parser.add_argument("--alpha", "-a", type=float, default=STEP_SIZE)
    parser.add_argument("--save", "-s", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    # If there's an output specified, then use the user specified output.  Otherwise, create file in the current
    # directory with the game's name.
    if args.output:
        file_name = args.output
    else:
        file_name = os.getcwd() + "/" + "space_invaders"

    load_file_path = None
    if args.loadfile:
        load_file_path = args.loadfile

    print('Cuda available?: ' + str(torch.cuda.is_available()))
    dqn(file_name, True, load_file_path)


if __name__ == '__main__':
    main()


