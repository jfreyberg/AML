import numpy as np
import copy
import sympy as sy
from pprint import pprint
from settings import *

class Test:
    def __init__(self, cost):
        self.cost = cost
        self.discount = 1

class Vaccine:
    def __init__(self, cost):
        self.cost = cost
        self.discount = 1
    
class Game:
    def __init__(self, NUM_PEOPLE):
        self.yearly_lockdown_capacity = 0

        self.susceptible = NUM_PEOPLE - 1
        self.infected = 1
        self.recovered = 0
        self.dead = 0
        self.quarantined = 0

        self.rec_duration = 2
        self.break_rate = 1/52
        
        x = sy.symbols('x')
        self.dead_rate = sy.solveset(sy.Eq(1-(1-x)**self.rec_duration,0.02), x).args[0]
        self.r_factor = 3

        self.inf_rate = self.r_factor/self.rec_duration
        self.rec_rate = 1/self.rec_duration
        
        self.lockdown_capacity = self.yearly_lockdown_capacity
        self.week = 0


        self.vaccinations_done = 0
        self.tests_done = 0
        self.known_infs = np.array([0,0]) #people in second week and people in first week
        self.money = NUM_PEOPLE
        self.total = NUM_PEOPLE
        self.test = Test(100000)
        self.vacc = Vaccine(100000)
        self.countermeasures = {"quarantine" : 0, "lockdown" : 0}
        self.state_history = [self.get_status()]
        self.action_history = []

    def update(self):
        current = np.array([copy.copy(self.susceptible),copy.copy(self.infected),copy.copy(self.recovered),copy.copy(self.dead),copy.copy(self.quarantined)])
        
        self.susceptible += -(self.inf_rate * current[0] * current[1])/self.total + self.break_rate*current[2] 
        self.infected += (self.inf_rate * current[0] * current[1])/self.total - self.rec_rate*current[1] - self.dead_rate*current[1]
        self.recovered += self.rec_rate*current[1] - self.break_rate*current[2] + self.rec_rate*current[4] 
        self.dead += self.dead_rate*current[1] + self.dead_rate*current[4]
        self.quarantined +=  - self.rec_rate*current[4] - self.dead_rate*current[4]

        self.week += 1
        if self.week % 52 == 0:
            self.lockdown_capacity = self.yearly_lockdown_capacity

        self.state_history.append(self.get_status())

    
    def get_status(self):
        if NORMALIZE_STATES:
            return [self.week/520, self.dead/1000000, self.quarantined/1000000, (self.test.cost)/1000, (self.vacc.cost)/1000, self.recovered/100000, self.vaccinations_done/15000, self.tests_done/300000]
        else:
            return [self.week, self.dead, self.quarantined, self.test.cost, self.vacc.cost, self.recovered, self.vaccinations_done, self.tests_done]

    def get_two_level_status(self):
        if len(self.state_history) > 1:
            return list(np.array([self.state_history[-1], self.state_history[-2]]).flatten())
        return list(np.array([self.state_history[-1], self.state_history[-1]]).flatten())

    def fulfill_actions(self, actions):
        # actions are given as list that is formatted as:
        # quarantine strategy[0,1,2,3], lockdown strategy[0,1,2], test budget[research, test ], vaccination budget[research, vacc]
        self.action_history.append(actions)
        self.vaccinations_done = 0
        self.tests_done = 0
        funcs = [self.set_quarantine, self.do_lockdown, self.do_test,self.do_vaccine]

        for func, action in zip(funcs, actions):
            func(action)
        
        self.update()

        
    def set_quarantine(self, type_=1):
        self.countermeasures["quarantine"] = type_


    def do_lockdown(self, type_):
        if (self.lockdown_capacity > type_):
            self.countermeasures["lockdown"] = type_
            self.lockdown_capacity -= type_
        else:
            self.countermeasures["lockdown"] = 0

        if(self.countermeasures["lockdown"] == 0):
            self.r_factor = 3
            self.inf_rate = self.r_factor/self.rec_duration
            self.rec_rate = 1/self.rec_duration

        if(self.countermeasures["lockdown"] == 1):
            self.r_factor = 1.5
            self.inf_rate = self.r_factor/self.rec_duration
            self.rec_rate = 1/self.rec_duration

        elif(self.countermeasures["lockdown"] == 2):
            self.r_factor = 1.2
            self.inf_rate = self.r_factor/self.rec_duration
            self.rec_rate = 1/self.rec_duration




    def do_test(self, test_budget):
        test_alloc, execute_alloc = test_budget
        research_alloc = 1- execute_alloc
        self.test.discount -= 2.0774509799928715e-08*(self.money*test_alloc*research_alloc)
        self.test.cost = np.max((100000**self.test.discount, 7))
        cost = self.test.cost
        num_people_to_test = np.min((self.total,np.floor(self.money*test_alloc*execute_alloc/cost))) * 4
        positives = num_people_to_test * self.infected/(self.susceptible + self.infected + self.recovered)
        self.known_infs = np.roll(self.known_infs,-1)
        self.known_infs[-1] = positives
        self.tests_done = num_people_to_test
        if(self.countermeasures["quarantine"] == 1):
            self.quarantined += positives*0.85
            self.infected -= positives*0.85
        elif(self.countermeasures["quarantine"] == 2):
            self.quarantined += positives*0.95
            self.infected -= positives*0.95
        elif(self.countermeasures["quarantine"] == 3):
            self.dead += positives
            self.infected -= positives



    def do_vaccine(self, vaccine_budget):
        vaccine_alloc, execute_alloc = vaccine_budget
        research_alloc = 1- execute_alloc
        self.vacc.discount -= 3.1549019599857427e-09*(self.money*vaccine_alloc*research_alloc)
        self.vacc.cost = np.max((100000**self.vacc.discount, 70))
        cost = self.vacc.cost
        num_people_to_vacc = np.floor(self.money*vaccine_alloc*execute_alloc/cost)
        if(num_people_to_vacc - self.susceptible < 0):
            self.recovered += num_people_to_vacc
            self.susceptible -= num_people_to_vacc
            self.vaccinations_done = num_people_to_vacc
        else:
            self.recovered += self.susceptible
            self.vaccinations_done = self.susceptible
            self.susceptible = 0







