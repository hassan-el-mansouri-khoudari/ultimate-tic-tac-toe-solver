import pygame
import math
import time
import pygame_menu

# Initializing Pygame
pygame.init()
Cross = 1
Circle = 2


# Game Setup
ROWS = 9 # ROWS == COLUMNS 
sleeptime = 0.3 # time between every move
nb_games = 10 #number of games given for algos flat ==> GRAVE to play for each legal move
budget = 600 # budget should be bigger than 514 
c = 128

##### Choose ALGO VS ALGO #####
##### Please choose which algo plays as Cross ! It has a big influence #####
##### algos = 'random', 'flat', 'UCB', 'UCT', 'RAVE', 'GRAVE', 'SequentialHalving', 'SHUSS' #####
algo1 = 'SHUSS'
algo2 = "GRAVE"
first_algo_play= Cross




# Screen
WIDTH = 800

win = pygame.display.set_mode((WIDTH, WIDTH))
pygame.display.set_caption("TicTacToe")



# Colors

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Images
gap = WIDTH // ROWS
X_IMAGE = pygame.transform.scale(pygame.image.load("Images/x.png"), (int(gap*0.85),int(gap*0.85)))
O_IMAGE = pygame.transform.scale(pygame.image.load("Images/o.png"), (int(gap*0.85), int(gap*0.85)))

# Fonts
END_FONT = pygame.font.SysFont('courier', 40)


import numpy as np
import random
import copy

# Constants

Empty = 0
Cross = 1
Circle = 2
Both = 3

    
class Board(object):
    def __init__(self):      # On suppose que Dx, Dy >= 3
        self.h = 0
        self.turn = Cross
        self.Dx = 3
        self.Dy = 3
        self.board = np.zeros((9,3,3))
        self.big_board = np.zeros((3,3))
        hashtable = []
        self.square = None
        for k in range (3):
            l = []
            for m in range (9) :
                l2=[]
                for i in range (3):
                    l1 = []
                    for j in range (3):
                        l1.append (random.randint (0, 2 ** 64))
                    l2.append (l1)
                l.append(l2)
            hashtable.append (l)
        self.hashTable = hashtable
        self.hashTurn = random.randint (0, 2 ** 64)
    def play (self, move):
        self.h = self.h ^ self.hashTable [move.form][move.square][move.x] [move.y]
        self.h = self.h ^ self.hashTurn 
        self.board [move.square][move.x] [move.y] = move.form
        if (move.form == Circle):
            self.turn = Cross
        else:
            self.turn = Circle
        self.square = move.x * 3 + move.y
        if len(self.legalMoves()) == 0:
            self.square = None
    def legalMoves(self):
        """
        Liste des coups autorisés
        """
        moves = []
        for square in range (0,9):
          for i in range (0, self.Dx):
              for j in range (0, self.Dy):
                  m = Move(self.turn,square,i,j)
                  if m.valid(self):
                    moves.append(m)
        return moves
    
    def check_bas_small(self,square,i,j):
      form = self.board[square][i][j]
      if form == self.board[square][i+1][j] and form == self.board[square][i+2][j] :
          return 1
      return 0
    def check_droite_small(self,square,i,j):
      form = self.board[square][i][j]
      if form == self.board[square][i][j+1] and form == self.board[square][i][j+2] :
          return 1
      return 0
    def check_diagonal_droite_small(self,square,i,j):
      form = self.board[square][i][j]
      if form == self.board[square][i+1][j+1] and form == self.board[square][i+2][j+2] :
          return 1
      return 0
    def check_diagonal_gauche_small(self,square,i,j):
      form = self.board[square][i][j]
      if form == self.board[square][i+1][j-1] and form == self.board[square][i+2][j-2] :
          return 1
      return 0
    


    def check_bas(self,i,j):
      f1,f2,f3 = self.big_board[i][j],self.big_board[i+1][j],self.big_board[i+2][j]
      f =  min(f1,f2,f3)
      if f1 ==3 : f1 = f
      if f2 ==3 : f2= f
      if f3 ==3 : f3= f
      if f1 == f2 and f2 == f3 and f1 != 0:
          return 1,f
      return 0,None

    def check_droite(self,i,j):
      f1,f2,f3 = self.big_board[i][j],self.big_board[i][j+1],self.big_board[i][j+2]
      f =  min(f1,f2,f3)
      if f1 ==3 : f1 = f
      if f2 ==3 : f2=f
      if f3 ==3 : f3= f
      if f1 == f2 and f2 == f3 and f1 != 0:
          return 1,f
      return 0,None

    def check_diagonal_droite(self,i,j):
      f1,f2,f3 = self.big_board[i][j],self.big_board[i+1][j+1] ,self.big_board[i+2][j+2]
      f =  min(f1,f2,f3)
      if f1 ==3 : f1 = f
      if f2 ==3 : f2=f
      if f3 ==3 : f3= f
      if f1 == f2 and f2 == f3 and f1 != 0:
          return 1,f
      return 0,None

    def check_diagonal_gauche(self,i,j):
      f1,f2,f3 = self.big_board[i][j],self.big_board[i+1][j-1] ,self.big_board[i+2][j-2]
      f =  min(f1,f2,f3)
      if f1 ==3 : f1 = f
      if f2 ==3 : f2=f
      if f3 ==3 : f3= f
      if f1 == f2 and f2 == f3 and f1 != 0:
          return 1,f
      return 0,None

    def score (self):
        """
        Renvoie le score, 1 si la croix gagne, 0 si le cercle gagne, -1 si la partie n'est pas finie, 0.5 si on a match nul
        """
        for square in range(9):
          for i in range(self.Dx):
              for j in range((self.Dy)):
                  if self.board[square][i][j] != Empty :
                      if self.Dx - i > 2 :
                          if self.check_bas_small(square,i,j) == 1 :
                                form = self.board[square][i][j]
                                x,y = square//3,square%3
                                if form == Circle and self.big_board[x][y] == Empty :
                                    self.big_board[x][y]  = Circle
                                elif form == Cross and self.big_board[x][y]  == Empty:
                                    self.big_board[x][y]  = Cross
                                elif (form == Circle and self.big_board[x][y]  == Cross) or ( form == Cross and self.big_board[x][y]  == Circle):
                                    self.big_board[x][y]  = Both

                      if self.Dy - j > 2 :
                          if self.check_droite_small(square,i,j) == 1 :
                                form = self.board[square][i][j]
                                x,y = square//3,square%3
                                if form == Circle and self.big_board[x][y] == Empty :
                                    self.big_board[x][y]  = Circle
                                elif form == Cross and self.big_board[x][y]  == Empty:
                                    self.big_board[x][y]  = Cross
                                elif (form == Circle and self.big_board[x][y]  == Cross) or ( form == Cross and self.big_board[x][y]  == Circle):
                                    self.big_board[x][y]  = Both

                      if self.Dx - i > 2 and self.Dy-j>2:
                          if self.check_diagonal_droite_small(square,i,j) == 1 :
                                form = self.board[square][i][j]
                                x,y = square//3,square%3
                                if form == Circle and self.big_board[x][y] == Empty :
                                    self.big_board[x][y]  = Circle
                                elif form == Cross and self.big_board[x][y]  == Empty:
                                    self.big_board[x][y]  = Cross
                                elif (form == Circle and self.big_board[x][y]  == Cross) or ( form == Cross and self.big_board[x][y]  == Circle):
                                    self.big_board[x][y]  = Both
                      
                      if self.Dx - i > 2 and j>1:
                          if self.check_diagonal_gauche_small(square,i,j) == 1 :
                                form = self.board[square][i][j]
                                x,y = square//3,square%3
                                if form == Circle and self.big_board[x][y] == Empty :
                                    self.big_board[x][y]  = Circle
                                elif form == Cross and self.big_board[x][y]  == Empty:
                                    self.big_board[x][y]  = Cross
                                elif (form == Circle and self.big_board[x][y]  == Cross) or ( form == Cross and self.big_board[x][y]  == Circle):
                                    self.big_board[x][y]  = Both

          for i in range(self.Dx):
              for j in range((self.Dy)):
                  if self.big_board[i][j] != Empty :
                      if self.Dx - i > 2 :
                          check,form = self.check_bas(i,j)
                          if  check== 1 :
                                if form == Circle :
                                    return 0
                                else :
                                    return 1

                      if self.Dy - j > 2 :
                          check,form = self.check_droite(i,j)
                          if  check== 1 :
                                if form == Circle :
                                    return 0
                                else :
                                    return 1

                      if self.Dx - i > 2 and self.Dy-j>2:
                          check,form = self.check_diagonal_droite(i,j)
                          if  check== 1 :
                                if form == Circle :
                                    return 0
                                else :
                                    return 1
                      
                      if self.Dx - i > 2 and j>1:
                          check,form = self.check_diagonal_gauche(i,j)
                          if  check== 1 :
                                if form == Circle :
                                    return 0
                                else :
                                    return 1

        l = self.legalMoves()
        if len (l) == 0:
              return 0.5
        return -1

    def terminal (self):
        """
        Si la partie est finie, renvoie True
        """
        if self.score () == -1:
            return False
        return True
    
    """
    Playout aléatoires
    """

    def playout (self):
        while (True):
            moves = self.legalMoves()
            if self.terminal():
                return self.score()
            n = random.randint (0, len (moves) - 1)
            self.play (moves [n])
            # print(self.board)

    def playoutAMAF(self, played):
        while(True):
            moves = []
            moves = self.legalMoves()
            if len(moves) == 0 or self.terminal():
                return self.score()
            n = random.randint(0, len(moves) - 1)
            played += [moves[n].code(self)]
            self.play(moves[n])

    
    
    

    def algo1_vs_algo2(self, algo1, algo2, first_algo_play=Cross, verbose_big=False,verbose_board=False):
        moves_list = []  

        algorithms = {'random': random_algo, 'flat': flat, 'UCB': UCB, 'UCT':BestMoveUCT, 'RAVE': BestMoveRAVE, 'GRAVE':BestMoveGRAVE,
                     'SequentialHalving': SequentialHalving, 'SHUSS' : SHUSS}
        
        while (True):
            moves = self.legalMoves()
            score = self.score()
            if verbose_big ==True:
                print(self.big_board)
            if verbose_board ==True:
                print(self.board)
            if score != -1 :
                return moves_list
            if self.turn == first_algo_play:
                move = algorithms[algo1](self)
                self.play(move)
                moves_list.append(move)
            else:
                move = algorithms[algo2](self)
                self.play(move)
                moves_list.append(move)



    
#     def PUCT_vs_UCB(self,k,first_algo_play = Cross):
#         while (True):
#             moves = self.legalMoves()
#             if self.terminal():
#                 return self.score()
#             if self.turn == first_algo_play:
#                 self.play(BestMovePUCT(self, k))
#             else:
#                 self.play(UCB(self, k))
#             #print(self.board) 


class Move(object):
    def __init__(self, form,square, x, y):
        self.form = form
        self.square = square
        self.x = x
        self.y = y
    def valid (self, board):
        """
        Le move est-il valide ?
        """
        if self.x >= board.Dx or self.y >= board.Dy or self.x < 0 or self.y < 0:
            return False
        if board.square != None and board.square != self.square :
            return False
        if board.board[self.square][self.x][self.y] != Empty :
            return False
        return True
    def code(self, board):
        """
        Code les moves
        """
        if self.form == Cross:
            return self.square*9+board.Dy*self.x + self.y
        else:
            return board.Dx*board.Dy*9+self.square*9+board.Dy*self.x + self.y



"""
RANDOM
"""
def random_algo(board):
    moves = board.legalMoves()
    n = random.randint (0, len (moves) - 1)
    return moves[n]

# Flat monte carlo

def flat(board):
    moves = board.legalMoves()
    bestScore = 0
    bestMove = moves [0]
    for m in range (len(moves)):
        sum = 0
        for i in range (nb_games):
            b = copy.deepcopy (board)
            b.play (moves [m])
            r = b.playout ()
            if board.turn == Circle:
                r = 1 - r
            sum = sum + r
        if sum > bestScore:
            bestScore = sum
            bestMove = moves [m]
    return bestMove

# UCB

def UCB (board):
    moves = board.legalMoves ()
    sumScores = [0.0 for x in range (len (moves))]
    nbVisits = [0 for x in range (len(moves))]
    for i in range (nb_games):
        bestScore = 0
        bestMove = 0
        for m in range (len(moves)):
            score = 1000000
            if nbVisits [m] > 0:
                 score = sumScores [m] / nbVisits [m] + 0.4 * np.sqrt (np.log (i) / nbVisits [m])
            if score > bestScore:
                bestScore = score
                bestMove = m
        b = copy.deepcopy (board)
        b.play (moves [bestMove])
        r = b.playout ()
        if board.turn == Circle:
            r = 1.0 - r
        sumScores [bestMove] += r
        nbVisits [bestMove] += 1
    bestScore = 0
    bestMove = moves [0]
    for m in range (len(moves)):
        score = nbVisits [m]
        if score > bestScore:
            bestScore = score
            bestMove = moves [m]
    return bestMove

"""
Fonctions utiles hors classes pour les algorithmes
"""

def look(Table, board):
    return Table.get(board.h, None)

def add(Table, board):
    Dx = board.Dx
    Dy = board.Dy
    MaxLegalMoves = 2*Dx*Dy*9
    nbplayouts = [0.0 for x in range(MaxLegalMoves)]
    nwins = [0.0 for x in range(MaxLegalMoves)]
    Table[board.h] = [0, nbplayouts, nwins]


def UCT(Table, board):
    if board.terminal():
        return board.score()
    t = look(Table, board)
    if t != None:
        bestValue = -10000000.0
        best = 0
        moves = board.legalMoves()
        for i in range(0, len(moves)):
            val = 10000000.0
            if t[1][i] > 0:
                Q = t[2][i] / t[1][i]
                if board.turn == Circle:
                    Q = 1 - Q
                val = Q + 0.4*np.sqrt(np.log(t[0])/t[1][i])
            if val > bestValue :
                bestValue = val
                best = i
        board.play(moves[best])
        res = UCT(Table, board)
        t[0] += 1
        t[1][best] += 1
        t[2][best] += res
        return res
    else:
        add(Table, board)
        return board.playout()

def BestMoveUCT(board):
    Table = {}
    for i in range(nb_games):
        b1 = copy.deepcopy(board)
        res = UCT(Table, b1)
    t = look(Table, board)
    moves = board.legalMoves()
    best = moves[0]
    bestValue = t[1][0]
    for i in range(1, len(moves)):
        if (t[1][i] > bestValue):
            bestValue = t[1][i]
            best = moves[i]
    return best

"""
AMAF
"""

def addAMAF(Table, board):
    Dx = board.Dx
    Dy = board.Dy
    MaxLegalMoves = 2*Dx*Dy*9
    MaxTotalLegalMoves = 2*Dx*Dy*9
    nbplayouts = [0.0 for x in range(MaxLegalMoves)]
    nwins = [0.0 for x in range(MaxLegalMoves)]
    nbplayoutsAMAF = [0.0 for x in range(MaxTotalLegalMoves)]
    nwinsAMAF = [0.0 for x in range(MaxTotalLegalMoves)]
    Table[board.h] = [1, nbplayouts, nwins, nbplayoutsAMAF, nwinsAMAF]
    
"""
RAVE
"""

def RAVE(Table, board, played):
    if (board.terminal()):
        return board.score()
    t = look(Table, board)
    if t!=None:
        bestValue = -10000000.0
        best = 0
        moves = board.legalMoves()
        bestCode = moves[0].code(board)
        for i in range(0, len(moves)):
            val = 1000000.0
            code = moves[i].code(board)
            # print(t[3])
            if t[3][code] > 0:
                beta = t[3][code] / (t[1][i] + t[3][code] + 1e-5 * t[1][i] * t[3][code])
                Q = 1
                if t[1][i] > 0:
                    Q = t[2][i] / t[1][i]
                    if board.turn == Circle:
                        Q = 1 - Q
                AMAF = t[4][code] / t[3][code]
                if board.turn == Circle:
                    AMAF = 1 - AMAF
                val = (1.0 - beta) * Q + beta * AMAF
            if val > bestValue:
                bestValue = val
                best = i
                bestCode = code
        board.play(moves[best])
        res = RAVE(Table, board, played)
        t[0] += 1
        t[1][best] += 1
        t[2][best] += res
        played.insert(0, bestCode)
        for k in range(len(played)):
            code = played[k]
            seen = False
            for j in range(k):
                if played[j] == code:
                    seen = True
            if not seen:
                t[3][code] += 1
                t[4][code] += res
        # played.insert(0, moves[best])
        return res
    else:
        addAMAF(Table, board)
        return board.playoutAMAF(played)



def BestMoveRAVE(board):
    Table = {}
    for i in range(nb_games):
        b1 = copy.deepcopy(board)
        res = RAVE(Table, b1, [])
    t = look(Table, board)
    moves = board.legalMoves()
    best = moves[0]
    bestValue = t[1][0]
    for i in range(1, len(moves)):
        if (t[1][i] > bestValue):
            bestValue = t[1][i]
            best = moves[i]

    ##Enlever les commentaires pour print les statistiques AMAF
    # print("t3")
    # print(t[3])
    # print("t4")
    # print(t[4])
    return best

#GRAVE

def GRAVE(Table, board, played, tref):
    if (board.terminal()):
        return board.score()
    t = look(Table, board)
    if t != None:
        tr = tref
        if t[0] > 50:
            tr = t
        bestValue = -100000.0
        best = 0
        moves = board.legalMoves()
        bestcode = moves[0].code(board)
        for i in range(0, len(moves)):
            val = 100000.0
            code = moves[i].code(board)
            if tr[3][code] > 0:
                beta = tr[3][code] / (t[1][i] + tr[3][code] + 1e-5*t[1][i] * t[3][code])
                Q = 1
                if t[1][i] > 0:
                    Q = t[2][i] / t[1][i]
                    if board.turn == Circle:
                        Q = 1 - Q
                AMAF = tr[4][code] / tr[3][code]
                if board.turn == Circle:
                    AMAF = 1 - AMAF
                val = (1.0 - beta) * Q + beta * AMAF
            if val > bestValue:
                bestValue = val
                best = i
                bestcode = code
        board.play(moves[best])
        played += [bestcode]
        res = GRAVE(Table, board, played, tr)
        t[0] += 1
        t[1][best] += 1
        t[2][best] += res
        for i in range(len(played)):
            code = played[i]
            seen = False
            for j in range(i):
                if played[j] == code:
                    seen = True
            if not seen:
                t[3][code] += 1
                t[4][code] += res
        return res
    else:
        addAMAF(Table, board)
        return board.playoutAMAF(played)

    
def BestMoveGRAVE(board):
    Table = {}
    for i in range(nb_games):
        t = look(Table, board)
        b1 = copy.deepcopy(board)
        res = GRAVE(Table, b1, [], t)
    t = look(Table, board)
    moves = board.legalMoves()
    best = moves[0]
    bestValue = t[1][0]
    for i in range(1, len(moves)):
        if (t[1][i] > bestValue):
            bestValue = t[1][i]
            best = moves[i]
    return best

#Sequential Halving

def SequentialHalving(board):
    Table = {}
    Dx = board.Dx
    Dy = board.Dy
    MaxLegalMoves = 2*Dx*Dy*9
    MaxTotalLegalMoves = 2*Dx*Dy*9
    moves = board.legalMoves()
    total = len(moves)
    nbplayouts = [0.0 for x in range(MaxTotalLegalMoves)]
    nbwins = [0.0 for x in range(MaxTotalLegalMoves)]
    while len(moves) > 1:
        for m in moves:
            for i in range(int(budget / (len(moves)*np.log2(total)))):
                b = copy.deepcopy(board)
                b.play(m)
                res = UCT(Table, b)
                nbplayouts[m.code(board)] += 1
                if board.turn == Cross:
                    nbwins[m.code(board)] += res
                else:
                    nbwins[m.code(board)] += 1.0 - res
        moves = bestHalf(board, moves, nbwins, nbplayouts)
    return moves[0]

def bestHalf(board, moves, nbwins, nbplayouts):
    half = []
    Dx = board.Dx
    Dy = board.Dy
    MaxLegalMoves = 2*Dx*Dy*9
    MaxTotalLegalMoves = 2*Dx*Dy*9
    notused = [True for x in range(MaxTotalLegalMoves)]
    for i in range(len(moves) // 2):
        best = -1.0
        bestMove = moves[0]
        for m in moves:
            code = m.code(board)
            if notused[code]:
                # mu = nbwins[code]
                # nbplayouts[code] += 1
                mu = nbwins[code] / nbplayouts[code]
                if mu > best:
                    best = mu
                    bestMove = m
        notused[bestMove.code(board)] = False
        half += [bestMove]
    return half


#SHUSS

def updateAMAF(t, played, res):
    for i in range(len(played)):
        code = played[i]
        seen = False
        for j in range(i):
            if played[j] == code:
                seen = True
        if not seen:
            t[3][code] += 1
            t[4][code] += res

def SHUSS(board):
    Table = {}
    Dx = board.Dx
    Dy = board.Dy
    MaxLegalMoves = 2*Dx*Dy*9
    MaxTotalLegalMoves = 2*Dx*Dy*9
    addAMAF(Table, board)
    t = look(Table, board)
    moves = board.legalMoves()
    total = len(moves)
    nbplayouts = [0.0 for x in range(MaxTotalLegalMoves)]
    nbwins = [0.0 for x in range(MaxTotalLegalMoves)]
    while len(moves) > 1:
        for m in moves:
            for i in range(int(budget / (len(moves) * np.log2(total)))):
                b = copy.deepcopy(board)
                b.play(m)
                played = [m.code(board)]
                res = GRAVE(Table, b, played, t)
                updateAMAF(t, played, res)
                nbplayouts[m.code(board)] += 1
                if board.turn == Cross:
                    nbwins[m.code(board)] += res
                else:
                    nbwins[m.code(board)] += 1.0 - res
        moves = bestHalfSHUSS(t, board, moves, nbwins, nbplayouts, c)
    return moves[0]

def bestHalfSHUSS(t, board, moves, nbwins, nbplayouts, c = 128):
    Dx = board.Dx
    Dy = board.Dy
    MaxLegalMoves = 2*Dx*Dy*9
    MaxTotalLegalMoves = 2*Dx*Dy*9
    half = []
    notused = [True for x in range(MaxTotalLegalMoves)]
    # c = 128 ##initialement 128
    for i in range(len(moves) //2):
        best = -1.0
        bestMove = moves[0]
        for m in moves:
            code = m.code(board)
            if notused[code]:
                AMAF = t[4][code] / t[3][code]
                if board.turn == Circle:
                    AMAF = 1 - AMAF
                mu = nbwins[code]/nbplayouts[code] + c*AMAF/nbplayouts[code]
                if mu > best:
                    best = mu
                    bestMove = m
        notused[bestMove.code(board)] = False
        half += [bestMove]
    return half




def draw_grid():
    gap = WIDTH // ROWS

    # Starting points
    x = 0
    y = 0

    for i in range(ROWS):
        x = i * gap
        if i % 3 == 0 and i !=0:
            pygame.draw.line(win, BLACK, (x, 0), (x, WIDTH), 3)
            pygame.draw.line(win, BLACK, (0, x), (WIDTH, x), 3)
        else : 
            pygame.draw.line(win, GRAY, (x, 0), (x, WIDTH), 3)
            pygame.draw.line(win, GRAY, (0, x), (WIDTH, x), 3)

def initialize_grid():
    dis_to_cen = WIDTH // ROWS // 2

    # Initializing the array
    #game_array = [[None, None, None], [None, None, None], [None, None, None]]
    game_array = [[None for i in range(ROWS)] for i in range(ROWS)]

    for i in range(len(game_array)):
        for j in range(len(game_array[i])):
            x = dis_to_cen * (2 * j + 1)
            y = dis_to_cen * (2 * i + 1)

            # Adding centre coordinates
            game_array[i][j] = (x, y, "", True)

    return game_array


def click(game_array, a, b):
    global x_turn, o_turn, images

    # Mouse position
    m_x, m_y = a, b

    for i in range(len(game_array)):
        for j in range(len(game_array[i])):
            x, y, char, can_play = game_array[i][j]

            # Distance between mouse and the centre of the square
            dis = math.sqrt((x - m_x) ** 2 + (y - m_y) ** 2)

            # If it's inside the square
            if dis < WIDTH // ROWS // 2 and can_play:
                if x_turn:  # If it's X's turn
                    images.append((x, y, X_IMAGE))
                    x_turn = False
                    o_turn = True
                    game_array[i][j] = (x, y, 'x', False)

                elif o_turn:  # If it's O's turn
                    images.append((x, y, O_IMAGE))
                    x_turn = True
                    o_turn = False
                    game_array[i][j] = (x, y, 'o', False)



def check_bas(game_array,i,j):
    form = game_array[i][j][2]
    if form == game_array[i+1][j][2] and form == game_array[i+2][j][2] :
        return 1
    return 0

def check_droite(game_array,i,j):
    form = game_array[i][j][2]
    if form == game_array[i][j+1][2] and form == game_array[i][j+2][2] :
        return 1
    return 0

def check_diagonal_droite(game_array,i,j):
    form = game_array[i][j][2]
    if form == game_array[i+1][j+1][2] and form == game_array[i+2][j+2][2] :
        return 1
    return 0

def check_diagonal_gauche(game_array,i,j):
    form = game_array[i][j][2]
    if form == game_array[i+1][j-1][2] and form == game_array[i+2][j-2][2] :
        return 1
    return 0


# Checking if someone has won
def has_won(game_array,score):
    if score == 0.5 or score == -1 :
        return False
    elif score == 1 :
        display_message( "1st algorithm 'X' has won!")
        return True
    else :
        display_message( "2nd algorithm 'O' has won!")
        return True



def has_drawn(game_array,score):
    if score != 0.5 :
        return False
    display_message("It's a draw!")
    return True


def display_message(content):
    pygame.time.delay(500)
    win.fill(WHITE)
    end_text = END_FONT.render(content, 1, BLACK)
    win.blit(end_text, ((WIDTH - end_text.get_width()) // 2, (WIDTH - end_text.get_height()) // 2))
    pygame.display.update()
    pygame.time.delay(3000)


def render():
    win.fill(WHITE)
    draw_grid()

    # Drawing X's and O's
    for image in images:
        x, y, IMAGE = image
        win.blit(IMAGE, (x - IMAGE.get_width() // 2, y - IMAGE.get_height() // 2))

    pygame.display.update()

"""

def main(list_moves,score):
    global x_turn, o_turn, images, draw

    images = []
    draw = False

    run = True

    x_turn = True
    o_turn = False

    game_array = initialize_grid()
    render()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        for move in list_moves :
            x = move.y*gap + gap//2  + move.square%3*gap*3
            y = move.x*gap + gap//2  + move.square//3*3*gap
            if x_turn:  # If it's X's turn
                images.append((x, y, X_IMAGE))
                x_turn = False
                o_turn = True
                game_array[move.x][move.y] = (x, y, 'x', False)

            elif o_turn:  # If it's O's turn
                images.append((x, y, O_IMAGE))
                x_turn = True
                o_turn = False
                game_array[move.x][move.y] = (x, y, 'o', False)

            render()

            time.sleep(sleeptime)


        if has_won(game_array,score) or has_drawn(game_array,score):
            run = False

while True:
    if __name__ == '__main__':


        pygame.init()
        surface = pygame.display.set_mode((int(WIDTH), int(WIDTH)))
        algo1 = 'random'
        algo2 = 'random'
        def set_difficulty(value, difficulty):
            # Do the job here !
            global algo1, algo2
            if difficulty == 1:
                algo1 = value[0][0]
            if difficulty == 2:
                algo2 = value[0][0]

        def start_the_game():
            board = Board()
            list_moves = board.algo1_vs_algo2(algo1, algo2, first_algo_play = first_algo_play, verbose_big=False,verbose_board=False)
            main(list_moves,board.score())

        menu = pygame_menu.Menu(int(WIDTH*0.8), int(WIDTH*0.8), 'Bienvenue !',
                            theme=pygame_menu.themes.THEME_DARK)


        menu.add_label('Note : Veuillez patienter quelques \n minutes après le lancement \n')
        menu.add_selector('Algorithme 1 :', [('random',1), ('flat',1), ('UCB',1), ('UCT',1), ('RAVE',1), ('GRAVE',1), ('SequentialHalving',1), ('SHUSS',1)], onchange=set_difficulty)
        menu.add_selector('Algorithme 2  :', [('random',2), ('flat',2), ('UCB',2), ('UCT',2), ('RAVE',2), ('GRAVE',2), ('SequentialHalving',2), ('SHUSS',2)], onchange=set_difficulty)

        menu.add_button('Commencer le jeu', start_the_game)
        menu.add_button('Quitter', pygame_menu.events.EXIT)

        menu.mainloop(surface)

        break
"""
    ####################################################################################################"
    # 
    # 

def render_algo1_vs_algo2(b, algo1, algo2, first_algo_play=Cross, verbose_big=False,verbose_board=False):
        
        algorithms = {'random': random_algo, 'flat': flat, 'UCB': UCB, 'UCT':BestMoveUCT, 'RAVE': BestMoveRAVE, 'GRAVE':BestMoveGRAVE,
                     'SequentialHalving': SequentialHalving, 'SHUSS' : SHUSS}

        global x_turn, o_turn, images, draw

        images = []
        draw = False

        run = True

        x_turn = True
        o_turn = False

        game_array = initialize_grid()
        render()

        while run:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()

        
            moves = b.legalMoves()
            score = b.score()
            move = None
            if verbose_big ==True:
                print(b.big_board)
            if verbose_board ==True:
                print(b.board)
            #if score != -1 :
            #    return moves_list
            if b.turn == first_algo_play:
                move = algorithms[algo1](b)
                b.play(move)
                #moves_list.append(move)
            else:
                move = algorithms[algo2](b)
                b.play(move)
                #moves_list.append(move)
            
            x = move.y*gap + gap//2  + move.square%3*gap*3
            y = move.x*gap + gap//2  + move.square//3*3*gap
            if x_turn:  # If it's X's turn
                images.append((x, y, X_IMAGE))
                x_turn = False
                o_turn = True
                game_array[move.x][move.y] = (x, y, 'x', False)

            elif o_turn:  # If it's O's turn
                images.append((x, y, O_IMAGE))
                x_turn = True
                o_turn = False
                game_array[move.x][move.y] = (x, y, 'o', False)

            render()
            time.sleep(0.5)

            if has_won(game_array,score) or has_drawn(game_array,score):
                run = False

"""
def play(score):
    global x_turn, o_turn, images, draw

    images = []
    draw = False

    run = True

    x_turn = True
    o_turn = False

    game_array = initialize_grid()
    render()

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()



        for move in list_moves :
            x = move.y*gap + gap//2  + move.square%3*gap*3
            y = move.x*gap + gap//2  + move.square//3*3*gap
            if x_turn:  # If it's X's turn
                images.append((x, y, X_IMAGE))
                x_turn = False
                o_turn = True
                game_array[move.x][move.y] = (x, y, 'x', False)

            elif o_turn:  # If it's O's turn
                images.append((x, y, O_IMAGE))
                x_turn = True
                o_turn = False
                game_array[move.x][move.y] = (x, y, 'o', False)

            render()

            time.sleep(sleeptime)


        if has_won(game_array,score) or has_drawn(game_array,score):
            run = False
"""

while True:
    if __name__ == '__main__':


        pygame.init()
        surface = pygame.display.set_mode((int(WIDTH), int(WIDTH)))
        algo1 = 'random'
        algo2 = 'random'
        def set_difficulty(value, difficulty):
            # Do the job here !
            global algo1, algo2
            if difficulty == 1:
                algo1 = value[0][0]
            if difficulty == 2:
                algo2 = value[0][0]

        def start_the_game():
            board = Board()
            render_algo1_vs_algo2(board, algo1, algo2, first_algo_play, verbose_big = False, verbose_board = False)

        menu = pygame_menu.Menu(int(WIDTH*0.8), int(WIDTH*0.8), 'Bienvenue !',
                            theme=pygame_menu.themes.THEME_DARK)


        menu.add_label('Note : Veuillez patienter quelques \n minutes après le lancement \n')
        menu.add_selector('Algorithme 1 :', [('random',1), ('flat',1), ('UCB',1), ('UCT',1), ('RAVE',1), ('GRAVE',1), ('SequentialHalving',1), ('SHUSS',1)], onchange=set_difficulty)
        menu.add_selector('Algorithme 2  :', [('random',2), ('flat',2), ('UCB',2), ('UCT',2), ('RAVE',2), ('GRAVE',2), ('SequentialHalving',2), ('SHUSS',2)], onchange=set_difficulty)

        menu.add_button('Commencer le jeu', start_the_game)
        menu.add_button('Quitter', pygame_menu.events.EXIT)

        menu.mainloop(surface)

        break