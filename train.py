from SIR_game_environment import * 
from pprint import pprint
from tqdm import tqdm
import numpy as np
from agent import Agent
from settings import *
import argparse
import sys
import os
from rewards import *
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Training script.')
parser.add_argument('--model')
args = parser.parse_args()

if not args.model:
    print('Please provide a model name (python3 train.py --model YOUR_MODEL_NAME)')
    sys.exit()


# create directories (if necessary)
if not os.path.exists('plots'):
    os.makedirs('plots')

if not os.path.exists('data'):
    os.makedirs('data')

# function to calculate running mean (use for plotting)
def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0)) 
    return (cumsum[N:] - cumsum[:-N]) / float(N)

# keep track of losses (for plotting)
lossesB = []
lossesT = []
lossesV = []


input_dim = len(Game(1).get_two_level_status()) + 1

B = Agent(name=args.model + '_b', training=True, input_dim = input_dim - 1, gamma=0.9, burnin=1000, epsilon=1, learning_rate=1e-3,
                 action_dim = len(ACTIONS), batch_size=64, eps_min=0.1, eps_dec=1e-8, replace=10, memory_size=1000)
T = Agent(name=args.model + '_t', training=True, input_dim = input_dim, gamma=0.9, burnin=1000, epsilon=1, learning_rate=1e-3,
                 action_dim = len(ACTIONS), batch_size=64, eps_min=0.1, eps_dec=5e-7, replace=10, memory_size=1000)
V = Agent(name=args.model + '_v', training=True, input_dim = input_dim, gamma=0.9, burnin=1000, epsilon=1, learning_rate=1e-3,
                 action_dim = len(ACTIONS), batch_size=64, eps_min=0.1, eps_dec=5e-7, replace=10, memory_size=1000)


OUTPUT = False


if not VACCINATION_ONLY:
    accumulated_rewardB = 0
    accumulated_rewardT = 0
accumulated_rewardV = 0
acB = 0

for game_nr in tqdm(range(1000000)):
    OUTPUT = False
    if game_nr % 50 == 0:
        OUTPUT = True
    g = Game(1000000)
    prev_state = g.get_two_level_status()
    prev_state.append(0.5)
    actionB = 0.5
    for i in range(WEEKS):
        quarantine = 2
        lockdown = 2

        if i % STEP_SIZE == 0:
            predict_from_this = g.get_two_level_status()
            if not VACCINATION_ONLY:
                acB, exploitedB = B.choose_action(predict_from_this, OUTPUT)
                if acB: 
                    actionB = ACTIONS[acB]
            predict_from_this.append(actionB)
            acV, exploitedV = V.choose_action(predict_from_this, OUTPUT)
            if not VACCINATION_ONLY:
                acT, exploitedT = T.choose_action(predict_from_this, OUTPUT)
        
        actionV = ACTIONS[acV]
        if not VACCINATION_ONLY:
            actionT = ACTIONS[acT]
        else:
            actionB = 0
            actionT = 0

        actions = [quarantine, lockdown, [actionB, actionT], [1-actionB, actionV]]
        g.fulfill_actions(actions)


        state = g.get_two_level_status()
        next_acB = 0
        if not VACCINATION_ONLY:
            next_acB, _ = B.choose_action(state, OUTPUT)
        next_actionB = ACTIONS[next_acB]
        state.append(next_actionB)


        if not VACCINATION_ONLY:
            rewardB = np.sum(compute_reward(state[:8], prev_state[:8], actions, [new_dead], REWARD_MULTIPLIERS))  
            accumulated_rewardB += rewardB

            rewardT = np.sum(compute_reward(state[:8], prev_state[:8], actions, [new_dead], REWARD_MULTIPLIERS))  
            accumulated_rewardT += rewardT

        rewardV = np.sum(compute_reward(state[:8], prev_state[:8], actions, [new_dead], REWARD_MULTIPLIERS))   
        accumulated_rewardV += rewardV

        if i % STEP_SIZE == 0:
            done = False
            if i == 520:
                done = True
            if not VACCINATION_ONLY:
                B.store_transition(prev_state[:16], acB, accumulated_rewardB, state[:16], done)
                T.store_transition(prev_state, acT, accumulated_rewardT, state, done)
            V.store_transition(prev_state, acV, accumulated_rewardV, state, done)
            prev_state = state

            if OUTPUT:     
                if not VACCINATION_ONLY:
                    print('REWARD B: {} (week: {})'.format(accumulated_rewardB, i))
                    print('REWARD T: {} (week: {})'.format(accumulated_rewardT, i))
                print('REWARD V: {} (week: {})'.format(accumulated_rewardV, i))
            
            if not VACCINATION_ONLY:
                accumulated_rewardB = 0
                accumulated_rewardT = 0
            accumulated_rewardV = 0

        if not VACCINATION_ONLY:
            lossB = B.learn()
            lossT = T.learn()
        lossV = V.learn()

        if lossV:
            if not VACCINATION_ONLY:
                lossesB.append(lossB)
                lossesT.append(lossT)
            lossesV.append(lossV)

    if game_nr % 10 == 0:
        plt.clf()
        if not VACCINATION_ONLY:
            T.save()
            B.save()
            plt.plot(running_mean(lossesB, 10000), label='B')
            plt.plot(running_mean(lossesB, 100000))
            plt.legend()
            plt.savefig("./plots/{}_{}_B.png".format(args.model, game_nr))
            plt.clf()
            plt.plot(running_mean(lossesT, 10000), label='T')
            plt.plot(running_mean(lossesT, 100000))
            plt.legend()
            plt.savefig("./plots/{}_{}_T.png".format(args.model, game_nr))
            plt.clf()
        V.save()
        print(V.epsilon)
        plt.plot(running_mean(lossesV, 10000), label='V')
        plt.plot(running_mean(lossesV, 100000))
        plt.legend()
        plt.savefig("./plots/{}_{}_V.png".format(args.model, game_nr))










