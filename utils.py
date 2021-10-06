
import numpy as np
from copy import copy
import matplotlib.pyplot as plt
import pandas as pd
from pprint import pprint
from agent import Agent
from SIR_game_environment import Game
from settings import *
from rewards import *

def show_results(games, names):
    if type(games) != list:
        games = [games]
        names = [names]
    data = []
    for G, name in zip(games, names):
        total = G.dead + G.infected + G.recovered + G.susceptible
        data.append({
            'name': name,
            'dead (%)': round(G.dead/total*100, 2),
            'infected (%)': round(G.infected/total*100, 2),
            'quarantined (%)': round(G.quarantined/total*100, 2),
            'recovered (%)': round(G.recovered/total*100, 2),
            'susceptible (%)': round(G.susceptible/total*100, 2)
        })

    df = pd.DataFrame(data)
    display(df)

class InfectionPlot:
    def __init__(self, w):
        self.weeks = range(w)

        self.Susceptible_otime = np.zeros(len(self.weeks))
        self.Infected_otime = np.zeros(len(self.weeks))
        self.Recovered_otime = np.zeros(len(self.weeks))
        self.Dead_otime = np.zeros(len(self.weeks))
        self.Quarantined_otime = np.zeros(len(self.weeks))

        if not OPTIMIZED_PLOTS:
            self.fig, self.ax = plt.subplots(3,1, figsize=(12,14))
        else:
            self.fig, self.ax = plt.subplots(2,1, figsize=(12,6), gridspec_kw={'height_ratios': [2, 1]})

        self.fig.tight_layout()

        self.sus_y  = []
        self.inf_y  =[]
        self.rec_y  = []
        self.dead_y = []
        self.quar_y = []
        self.known_y = []
        self.ax[0].set_xlim(self.weeks[0],self.weeks[-1])
        self.ax[0].set_ylim(0,1)
        self.ax[0].set_xlabel('weeks')
        self.ax[0].set_ylabel('fraction of population')

        self.spendings_test_research = []
        self.spendings_test_exec = []
        self.spendings_vacc_research = []
        self.spendings_vacc_exec = []


        self.test_cost_y  = []
        self.vacc_cost_y = []
        if not OPTIMIZED_PLOTS:
            self.ax[1].set_xlim(self.weeks[0],self.weeks[-1])
            self.ax[1].set_ylim(0,120000)
            #self.ax[1].set_yscale("symlog")

            self.ax[2].set_xlim(self.weeks[0],self.weeks[-1])
            self.ax[2].set_ylim(0,1)
            self.ax[2].grid()
    
        else:

            self.ax[1].set_ylabel('fraction of bduget')
            self.ax[1].set_xlim(self.weeks[0],self.weeks[-1])
            self.ax[1].set_ylim(0,1)
            self.ax[1].grid()

    def update_line(self, G, actions):
        self.sus_y.append(G.susceptible)
        self.inf_y.append(G.infected)
        self.rec_y.append(G.recovered)
        self.dead_y.append(G.dead)
        self.quar_y.append(G.quarantined)
        self.known_y.append(G.known_infs.sum())

        self.test_cost_y.append(G.test.cost)
        self.vacc_cost_y.append(G.vacc.cost)

        test_alloc = actions[2][0]
        vacc_alloc = actions[3][0]

        test_exec = test_alloc * actions[2][1]
        test_research = test_alloc * (1-actions[2][1])

        vacc_exec = vacc_alloc * actions[3][1]
        vacc_research = vacc_alloc * (1-actions[3][1])

        self.spendings_test_research.append(test_research)
        self.spendings_test_exec.append(test_exec)
        self.spendings_vacc_research.append(vacc_research)
        self.spendings_vacc_exec.append(vacc_exec)


    def show(self):

        self.ax[0].plot(self.weeks, np.array(self.sus_y)/1000000, label="susceptible")
        self.ax[0].plot(self.weeks, np.array(self.inf_y)/1000000, label="infected")
        self.ax[0].plot(self.weeks, np.array(self.rec_y)/1000000, label="recovered")
        self.ax[0].plot(self.weeks, np.array(self.dead_y)/1000000, label="dead")
        self.ax[0].plot(self.weeks, np.array(self.quar_y)/1000000, label="quarantined")
        self.ax[0].plot(self.weeks, np.array(self.known_y)/1000000, label="tested")
        self.ax[0].legend(loc='upper right', ncol=6)

        if not OPTIMIZED_PLOTS:
            self.ax[1].plot(self.weeks, self.test_cost_y, label="Test Cost")
            self.ax[1].plot(self.weeks, self.vacc_cost_y, label="Vacc Cost")
            self.ax[1].legend()


            self.ax[2].stackplot(self.weeks, self.spendings_test_research, self.spendings_test_exec, self.spendings_vacc_research, self.spendings_vacc_exec, labels=ACTION_LABELS, colors=ACTION_COLORS)
            self.ax[2].legend(bbox_to_anchor=(0.5, 1.15), loc='upper center', ncol=len(ACTION_LABELS))
        else:
            self.ax[1].stackplot(self.weeks, self.spendings_test_research, self.spendings_test_exec, self.spendings_vacc_research, self.spendings_vacc_exec, labels=ACTION_LABELS, colors=ACTION_COLORS)
            self.ax[1].legend(bbox_to_anchor=(0.5, -0.4), loc='center', ncol=len(ACTION_LABELS))

        plt.show()

def agent_simulation(name, types, nr_weeks=WEEKS, output=False, population=1000000):

    if "B" in types:
        B = Agent(name=name + '_b', training=False)
    if "T" in types:
        T = Agent(name=name + '_t', training=False)
    if "V" in types:
        V = Agent(name=name + '_v', training=False)

    print_name = '{}, gamma: {}, learning_rate: {}'.format(*V.get_print_data())
    actionB = 0

    G = Game(population)
    P = InfectionPlot(nr_weeks)
    
    for i in range(nr_weeks):
        quarantine_strategy = 2
        lockdown = 2

        if i % STEP_SIZE == 0:
            predict_from_this = G.get_two_level_status()
            
            if "B" in types:
                acB, _ = B.choose_action(predict_from_this, output)
                if acB: 
                    actionB = ACTIONS[acB]
            predict_from_this.append(actionB)

            if "V" in types:
                acV, _ = V.choose_action(predict_from_this, output)

            if "T" in types:
                acT, _ = T.choose_action(predict_from_this, output)
        

        actionV = 0
        if "V" in types:
            actionV = ACTIONS[acV]
        actionT = 0
        if "T" in types:
            actionT = ACTIONS[acT]

        test_alloc, test_research_alloc, test_execute_alloc, vaccine_alloc, vaccine_research_alloc, vaccine_execute_alloc = (actionB, 1-actionT, actionT, 1-actionB, actionV,1-actionV)


        test_budget =  np.array([test_alloc,test_execute_alloc])
        vaccine_budget =  np.array([vaccine_alloc,vaccine_execute_alloc])

        actions = [quarantine_strategy, lockdown, test_budget, vaccine_budget]
        G.fulfill_actions(actions)
        P.update_line(G, actions)

    P.show()
    return G, print_name










