import numpy 

ACTIONS = {'U': (-1, 0), 'D': (1, 0), 'L': (0, -1), 'R': (0, 1)}


class Agent(object):
    def __init__(self, states, alpha=0.15, random_factor=0.2):  # 80% explore, 20% exploit
        self.state_history = [((0, 0), 0)]  # state, reward
        self.alpha = alpha
        self.random_factor = random_factor
        self.G = {}
        self.init_reward(states,0)
        #print(self.G)

    def init_reward(self,nim_rows,row):
        if row>=len(nim_rows):
            return
        for k in range(0,nim_rows[row]+1):
            new_state = nim_rows[:row] + [nim_rows[row] - k] + nim_rows[row+1:]
            #print(new_state)
            self.G[tuple(new_state)] = numpy.random.uniform(low=1.0, high=0.1)
            self.init_reward(new_state,row+1)

    def choose_action(self, state, allowedMoves):
        maxG = -10e15
        next_move = None
        randomN = numpy.random.random()
        if randomN < self.random_factor:
            # if random number below random factor, choose random action
            next_move = numpy.random.choice(allowedMoves)
        else:
            # if exploiting, gather all possible actions and choose one with the highest G (reward)
            for action in allowedMoves:
                new_state = tuple([sum(x) for x in zip(state, ACTIONS[action])])
                if self.G[new_state] >= maxG:
                    next_move = action
                    maxG = self.G[new_state]

        return next_move

    def update_state_history(self, state, reward):
        self.state_history.append((state, reward))

    def learn(self):
        target = 0

        for prev, reward in reversed(self.state_history):
            self.G[prev] = self.G[prev] + self.alpha * (target - self.G[prev])
            target += reward

        self.state_history = []

        self.random_factor -= 10e-5  # decrease random factor each episode of play
