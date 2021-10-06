
# reward functions

def cap_reward(r):
    if r > 0:
        return min(1.0, r)
    return 0

def new_dead(state, prev_state, actions):
    return - (state[1] - prev_state[1]) * 10e3 # people who died (negative!)


def test_cost_reduction(state, prev_state, actions):
    return cap_reward(prev_state[3] - state[3]) # test cost reduction

def vacc_cost_reduction(state, prev_state, actions):
    return cap_reward(prev_state[4] - state[4]) # vacc cost reduction


def punish_invalid_vacc_reduction(state, prev_state, action):
    if prev_state[4] == 0.0: # vacc cost reduction is impossible
        if action[3][1] != 1:
            return -(1-action[3][1])
    return 0


def recovered_persons(state, prev_state, actions):
    return state[5] - prev_state[5]


def vaccinations_done(state, prev_state, actions):
    return state[6]

def tests_done(state, prev_state, actions):
    return state[7]

def compute_reward(state, prev_state, action, funcs, factors):
    res = []
    for func, factor in zip(funcs, factors):
        res.append(func(state, prev_state, action) * factor)
    return res