import time, random
from Yinsh.yinsh_model import YinshGameRule
import numpy as np
import pickle


THINKTIME = 0.95
RING_POS = [(5,5), (6,5), (4,6), (7,5), (5,6), (4,5), (5,4), (6,4), (5,7), (5,8)]

class myAgent():

    def __init__(self,_id):
        self.id = _id 
        self.gameRule = YinshGameRule(2)
        self.weight = [0.9, -1, -0.9, 0.33, 0.05, -0.03]
        self.count = 0
        self.startTime = None

    def GetActions(self, state):
        return self.gameRule.getLegalActions(state, self.id)
    
    def GetOppActions(self, state):
        if self.id == 1:
            opp_id = 0
        elif self.id == 0:
            opp_id = 1
        return self.gameRule.getLegalActions(state, opp_id)

    def DoAction(self, state, action):
        self_score = state.agents[self.id].score
        state = self.gameRule.generateSuccessor(state, action, self.id)
        return state.agents[self.id].score - self_score

    def DoOppAction(self, state, action):
        opp_state = pickle.loads(pickle.dumps(state, -1))
        if self.id == 1:
            opp_id = 0
        elif self.id == 0:
            opp_id = 1
        opp_score = state.agents[opp_id].score
        opp_state = self.gameRule.generateSuccessor(opp_state, action, opp_id)
        return opp_state.agents[opp_id].score - opp_score

    def CalCounter(self, state, colour):
        sum = 0
        for i in range(len(state.board)):
            sum += str(state.board[i]).count(str(colour))
        return sum/85

    def HoriLine(self, state):
        return pickle.loads(pickle.dumps(state, -1)).board

    def VerLine(self, state):
        trans_state = np.transpose(pickle.loads(pickle.dumps(state, -1)).board)
        return trans_state
    
    def DiagLine(self, state):
        diag_board = []
        for j in range(len(state.board)):
            diag_board.append([state.board[i][j-i] for i in range(j+1)])
        for j in range(len(state.board), 2*len(state.board)-1):
            diag_board.append([state.board[i][j-i] for i in range(j-len(state.board)+1, len(state.board))])
        diag_board = diag_board[6:15]
        return diag_board

    def CalPoint(self, state, c):       
        self_c = c
        if self_c == 2:
            opp_c = 4
        elif self_c == 4:
            opp_c = 2 

        board_line = []
        hori_line = self.HoriLine(state)
        ver_line = self.VerLine(state)
        diag_line = self.DiagLine(state)

        for i in range(len(hori_line)):
            board_line.append(hori_line[i])
        for i in range(len(ver_line)):
            board_line.append(ver_line[i])
        for i in range(len(diag_line)):
            board_line.append(diag_line[i])
        count = 0
        
        for i in range(len(board_line)):   
            board_list = list(board_line[i])  
            e_list = [[c, c, c, c, c-1], [c, c, c, c-1, c], [c, c, c-1, c, c], [c, c-1, c, c, c], 
                    [c-1, c, c, c, c], [c, c, c, c - 1, opp_c, 0],[0, opp_c, c - 1, c, c, c], 
                    [c, c, c - 1, opp_c, opp_c, 0], [0, opp_c, opp_c, c - 1, c, c]]
            e2_list = [[c, c, c, c - 1, 0], [0, c - 1, c, c, c]]
            e3_list = [[c - 1, c - 1, c, c, c],
                    [c-1,c,c-1,c,c], [c-1,c,c,c-1,c],[c-1,c,c,c,c-1],[c,c-1,c-1,c,c],
                    [c,c-1,c,c-1,c],[c,c-1,c,c,c-1],[c,c,c-1,c-1,c],[c,c,c-1,c,c-1],
                    [c,c,c,c-1,c-1]]
        
            if board_list.count(c) >= 2:
                for j in range(len(e_list)):
                    if any([e_list[j]==board_list[m:m+len(e_list[j])] for m in range (0,len(board_list)-len(e_list[j]) + 1)]):
                        count += 1
            if board_list.count(c) >= 3 and board_list.count(c-1) >= 2:
                for j in range(len(e3_list)):
                    if any([e3_list[j]==board_list[m:m+len(e3_list[j])] for m in range (0,len(board_list)-len(e3_list[j]) + 1)]):
                        count += 0.5
            if board_list.count(c) >= 3:
                for j in range(len(e2_list)):
                    if any([e2_list[j]==board_list[m:m+len(e2_list[j])] for m in range (0,len(board_list)-len(e2_list[j]) + 1)]):
                        count += 0.3    
        return count

    def CalFeatures(self, state, action):
        if self.id == 1:
            opp_id = 0
        elif self.id == 0:
            opp_id = 1
        features = []
        self_colour = 2 * (self.id + 1)
        opp_colour = 2 * ((1 - self.id) + 1)
        curr_state = pickle.loads(pickle.dumps(state, -1))
        next_state = pickle.loads(pickle.dumps(state, -1))
        self_score = self.DoAction(next_state, action)
     
        opp_next_state = pickle.loads(pickle.dumps(next_state, -1))
        opp_next_score = self.CalOppScore(opp_next_state)
        opp_curr_score = next_state.agents[opp_id].score

        features.append(self_score) 
        features.append(opp_curr_score) 
        features.append(opp_next_score) 
   
        features.append(self.CalCounter(next_state, self_colour) - self.CalCounter(next_state, opp_colour))
        features.append(self.CalPoint(next_state, self_colour))
        features.append(self.CalPoint(next_state, opp_colour))
        
        return features

    def CalOppScore(self, state):
        opp_state = pickle.loads(pickle.dumps(state, -1))
        opp_legal_actions = self.GetOppActions(opp_state)
        opp_next_state = pickle.loads(pickle.dumps(state, -1))
        for action in opp_legal_actions:   
            opp_score = self.DoOppAction(opp_next_state, action)
            if opp_score != 0:
                return opp_score
        return 0

    def SelectAction(self, actions, gameState):
        
        self.count += 1
        start_time = time.time()
        selected_Q_value = -1000000
        selected_action = random.choice(actions)
        current_state = pickle.loads(pickle.dumps(gameState, -1))
        legal_actions = self.GetActions(current_state)
        
        if self.count < 6:
            for action in legal_actions:
                for pos in RING_POS:
                    if action['place pos'] == pos:
                        return action
            return random.choice(legal_actions)

        for action in legal_actions:
            if time.time()-start_time <= THINKTIME:
                features = self.CalFeatures(pickle.loads(pickle.dumps(gameState, -1)), action)
                Q_value = np.dot(features, self.weight)
                if Q_value > selected_Q_value:
                    selected_Q_value =  Q_value
                    selected_action = action
            else:
                print("TIME OUT!!")
                break
        return selected_action