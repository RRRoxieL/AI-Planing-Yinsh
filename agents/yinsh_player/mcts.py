
# IMPORTS AND CONSTANTS ----------------------------------------------------------------------------------------------#


from Yinsh.yinsh_model import YinshGameRule
from template import Agent
import numpy as np
import random
import pickle
import time


THINKTIME = 0.95
GAMMA = 0.9
EXPLORATION_WEIGHT = 2
STATE_WEIGHT = 150
STEPLIMIT = 2

V_LINE_POS = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7), (0, 8), (0, 9)]
H_D_LINE_POS = [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5), (6, 5), (7, 5), (8, 5), (9, 5)]


# CLASS DEFINED FOR TREE NODE ----------------------------------------------------------------------------------------------#


class Node:
    def __init__(self, state, action):
        self.action = action   # action that generated this node
        self.state = state
        self.parent = None
        self.children = {}
        self.value = 0
        self.visited = 0


class myAgent(Agent):
    def __init__(self, _id):
        self.id = _id
        self.gameRule = YinshGameRule(2)
        self.startTime = None
        self.allLine = self.GetAllLine()
        self.curr = 0
        self.oppo = 0

    def GetAllLine(self):
        h_line = [self.gameRule.positionsOnLine(point, 'h') for point in H_D_LINE_POS]
        d_line = [self.gameRule.positionsOnLine(point, 'd') for point in H_D_LINE_POS]
        v_line = [self.gameRule.positionsOnLine(point, 'v') for point in V_LINE_POS]
        return h_line + d_line + v_line

    def calReward(self, state):
        simulationReward = 0
        for line in self.allLine:
            simulationReward += self.calPoint(state.board, line, self.id)
        for line in self.allLine:
            simulationReward -= self.calPoint(state.board, line, 1-self.id)
        statetReward = (self.gameRule.calScore(state, self.id) - self.curr) - 0.8 * (self.gameRule.calScore(state, 1- self.id) - self.oppo)
        reward = statetReward * STATE_WEIGHT + simulationReward
        return reward

    def calPoint(self, gameBoard, line, agentID):
        point = 0
        factor = 1
        for pos in line:
            pos_num = gameBoard[pos]
            if pos_num == 5:
                continue
            elif pos_num == agentID * 2 + 1:
                point += factor * 0.5
                factor *= 2
            elif pos_num == agentID * 2 + 2:
                point += factor
                factor *= 2
            else:
                factor =1
            if factor > 16:
                factor = 4
        return point

    "PRIOR WORK"
    def BuildTree(self, gameState, actions):
        rootNode = Node(gameState, None)
        dumps = pickle.dumps(gameState)
        for action in actions:
            tempState = pickle.loads(dumps)
            tempState = self.gameRule.generateSuccessor(tempState, action, self.id)
            childNode = Node(tempState, action)
            childNode.parent = rootNode
            rootNode.children[childNode] = 0
        return rootNode

    def Selection(self, rootNode):
        maxUcb = -float('inf')
        selectedNode = None
        childrenList = list(rootNode.children.keys())
        np.random.shuffle(childrenList)
        for child in childrenList:
            if rootNode.children[child] == 0:
                selectedNode =  child
                break
            ucb = child.value + EXPLORATION_WEIGHT * np.sqrt(2 * np.log(rootNode.visited) / rootNode.children[child])
            if ucb > maxUcb:
                selectedNode = child
                maxUcb = ucb
        rootNode.children[selectedNode] +=1
        rootNode.visited += 1
        return selectedNode


    def Simulation(self, selectedNode):
        selectedNode.visited += 1
        step = 0
        tempState = pickle.loads(pickle.dumps(selectedNode.state))
        while step < STEPLIMIT:
            action = random.choice(self.gameRule.getLegalActions(tempState, self.id)) 
            tempState = self.gameRule.generateSuccessor(tempState,action,self.id)
            step += 1
        reward = self.calReward(tempState) * GAMMA
        selectedNode.value = (selectedNode.value * (selectedNode.visited - 1) + reward) / selectedNode.visited


    def BackPropagation(self, rootNode, startNode):
        tempNode = startNode
        length = 1
        while tempNode != rootNode:
            parentNode = tempNode.parent
            parentNode.value = max(tempNode.value * pow(GAMMA, length), parentNode.value)
            tempNode = parentNode
            length += 1

    def SelectAction(self, actions, gameState):
        self.startTime = time.time()
        self.curr = self.gameRule.calScore(gameState,self.id)
        self.oppo = self.gameRule.calScore(gameState, 1-self.id)

        count = 0
        rootNode = self.BuildTree(gameState, actions)

        "MCTS ITERATION"
        while time.time() - self.startTime <= THINKTIME:
            selectedNode = self.Selection(rootNode)
            self.Simulation(selectedNode)
            self.BackPropagation(rootNode, selectedNode)
            count += 1

        "SREACHING FOR BEST ACTION"
        maxValue = -float('inf')
        maxNode = None
        for child in rootNode.children.keys():
            if child.value > maxValue:
                maxValue = child.value
                maxNode = child

        return maxNode.action 
