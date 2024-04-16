# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        from util import manhattanDistance

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        score = successorGameState.getScore()

        foodList = newFood.asList()
        if foodList:
            closestFoodDist = min([manhattanDistance(newPos, foodPos) for foodPos in foodList])
            score += 1.0 / closestFoodDist

        # Adjust the score based on ghost positions and scared times
        ghostDist = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]
        for dist, scaredTime in zip(ghostDist, newScaredTimes):
            if scaredTime > 0:
                score += 10 - dist
            else:
                if dist < 2:
                    score -= 100
        return score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(agent, depth, gameState):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None

            if agent == 0:  # Pacman
                return maxAgent(agent, depth, gameState)
            else:  # Ghosts
                return minAgent(agent, depth, gameState)

        def maxAgent(agent, depth, gameState):
            maxEval = float("-inf"), None
            for action in gameState.getLegalActions(agent):
                nextGameState = gameState.generateSuccessor(agent, action)
                eval, _ = minimax((agent + 1) % gameState.getNumAgents(),
                                  depth - 1 if agent == gameState.getNumAgents() - 1 else depth, nextGameState)
                maxEval = max(maxEval, (eval, action), key=lambda x: x[0])
            return maxEval

        def minAgent(agent, depth, gameState):
            minEval = float("inf"), None
            for action in gameState.getLegalActions(agent):
                nextGameState = gameState.generateSuccessor(agent, action)
                eval, _ = minimax((agent + 1) % gameState.getNumAgents(),
                                  depth - 1 if agent == gameState.getNumAgents() - 1 else depth, nextGameState)
                minEval = min(minEval, (eval, action), key=lambda x: x[0])
            return minEval

        _, action = minimax(0, self.depth, gameState)
        return action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def alphaBeta(agentIndex, depth, gameState, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == 0:
                return self.evaluationFunction(gameState), None

            numAgents = gameState.getNumAgents()
            if agentIndex == 0:  # Pacman
                return maxVal(agentIndex, depth, gameState, alpha, beta)
            else:  # Ghost
                return minVal(agentIndex, depth, gameState, alpha, beta)

        def maxVal(agentIndex, depth, gameState, alpha, beta):
            v = float("-inf"), None
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                eval, _ = alphaBeta((agentIndex + 1) % gameState.getNumAgents(),
                                    depth - 1 if (agentIndex + 1) % gameState.getNumAgents() == 0 else depth, successorState,
                                    alpha, beta)
                if eval > v[0]:
                    v = eval, action
                if v[0] > beta:
                    return v  # prune branch
                alpha = max(alpha, v[0])
            return v

        def minVal(agentIndex, depth, gameState, alpha, beta):
            v = float("inf"), None
            for action in gameState.getLegalActions(agentIndex):
                successorState = gameState.generateSuccessor(agentIndex, action)
                eval, _ = alphaBeta((agentIndex + 1) % gameState.getNumAgents(),
                                    depth - 1 if (agentIndex + 1) % gameState.getNumAgents() == 0 else depth, successorState,
                                    alpha, beta)
                if eval < v[0]:
                    v = eval, action
                if v[0] < alpha:
                    return v
                beta = min(beta, v[0])
            return v

        alpha = float("-inf")
        beta = float("inf")
        _, action = alphaBeta(0, self.depth, gameState, alpha, beta)
        return action
class ExpectimaxAgent(MultiAgentSearchAgent):
    def getAction(self, gameState):
        def tState(state, depth):
            return state.isWin() or state.isLose() or depth == 0

        def max(state, depth):
            if tState(state, depth):
                return self.evaluationFunction(state), None
            value, bestAction = float("-inf"), None
            for action in state.getLegalActions(0):
                nextState = state.generateSuccessor(0, action)
                nextValue, _ = stra(1, depth, nextState)
                if value < nextValue:
                    value, bestAction = nextValue, action
            return value, bestAction

        def expect(state, agentInd, depth):
            if tState(state, depth):
                return self.evaluationFunction(state), None
            total, actions = 0, state.getLegalActions(agentInd)
            for action in actions:
                nextState = state.generateSuccessor(agentInd, action)
                nextValue, _ = stra((agentInd + 1) % state.getNumAgents(), depth - 1 if agentInd == state.getNumAgents() - 1 else depth, nextState)
                total += nextValue
            return total / len(actions) if actions else self.evaluationFunction(state), None

        def stra(agentInd, depth, state):
            if agentInd == 0:
                return max(state, depth)
            else:
                return expect(state, agentInd, depth)

        return stra(0, self.depth, gameState)[1]



def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def betterEvaluationFunction(currentGameState):

    pacmanP = currentGameState.getPacmanPosition()
    foodP = currentGameState.getFood().asList()
    ghostStates = currentGameState.getGhostStates()
    capsules = currentGameState.getCapsules()
    currScore = currentGameState.getScore()

    def getReciprocalDistance(items):
        distances = [manhattanDistance(pacmanP, item) for item in items]
        return min(distances) if distances else float('inf')

    def evaluateGhostProximity(ghostStates):
        scoreAdj = 0
        for ghost in ghostStates:
            distance = manhattanDistance(pacmanP, ghost.getPosition())
            if ghost.scaredTimer:  # If the ghost is scared
                scoreAdj += 50 / (distance + 1)
            else:
                # Penalize being too close to a non-scared ghost
                if distance <= 2:
                    scoreAdj -= 200 / (distance + 1)
        return scoreAdj

    nearestDst = getReciprocalDistance(foodP)
    foodScore = 1.0 / nearestDst if nearestDst != float('inf') else 0

    ghostScoreAdj = evaluateGhostProximity(ghostStates)

    capsuleScore = -2 * len(capsules)

    return currScore + foodScore + ghostScoreAdj + capsuleScore

# Abbreviation
better = betterEvaluationFunction
