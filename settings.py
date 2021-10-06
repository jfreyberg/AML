import numpy as np
# settings

DATA_DIRECTORY = 'data'
WEEKS = 520
ACTIONS = np.linspace(0,1,11)
REWARD_MULTIPLIERS = [1]
STEP_SIZE = 13
ACTION_LABELS = ['test-research', 'test-execution', 'vacc-research', 'vacc-execution']
ACTION_COLORS = ['deepskyblue','dodgerblue', 'limegreen', 'seagreen']
NORMALIZE_STATES = True
OPTIMIZED_PLOTS = True
VACCINATION_ONLY = False