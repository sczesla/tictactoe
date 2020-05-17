import numpy as np
import ttt
import argparse

from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
import keras

import re
from abc import abstractmethod
import itertools

class Player:
    
    def __init__(self, pl, name):
        self.pl = self._checkPlayer(pl)
        self.name = self._checkName(name)

    @staticmethod
    def _checkPlayer(p):
        """
        Check whether player is 1 (crosses) or 4 (circles)
        """
        if not p in [1,4]:
            raise(Exception("Player must be one or four"))
        return p

    @staticmethod
    def _checkName(name):
        """
        Check whether name is alphanumeric
        """
        if not re.match(r'^\w+$', name):
            raise(Exception("Player name must be alphanumeric plus underscore."))
        return name

    @abstractmethod
    def initGame(self):
        pass

    @abstractmethod
    def move(self):
        pass
    
    @abstractmethod
    def update(self):
        pass
    
    def __str__(self):
        return self.name


class RP(Player):
    """
    Random player
    
    Parameters
    ----------
    pl : int
        1 or 4 (crosses or circled)
    name : string
        Player name
    """
    
    def __init__(self, pl, name="RP"):
        super().__init__(pl, name)
        self.gamesplayed = 0

    def initGame(self):      
        """ Start new game """
        pass
        
    def move(self, t, **kwargs):
        """
        Choose a random (valid) move
        
        Returns
        -------
        Field number : int
            Field to be occupied
        """
        # Possible moves
        hm, pms = t.hypoMoves(self.pl)
        # Select random move
        move = np.random.randint(0, len(pms))        
        return pms[move]    
    
    def update(self, *args, **kwargs):
        """
        Game finished. Use reward, r, to update policy
        """
        self.gamesplayed += 1
    
    
class QP_nn(Player):
    """
    Q player (Neural network)
    
    Parameters
    ----------
    pl : int
        1 or 4 (crosses or circled)
    ri : boolean, optional
        If True, random initialization of Q
    dr : float
        Discount rate
    lr : float
        Learning rate
    eret : float
        Decay time of exploration rate
    """
    def __init__(self, pl, ri=True, name="Qnn_Player", eret=10000, lr=0.01):
        
        super().__init__(pl, name)
        
        self.policyupdates = 0
        self.eret = eret
        # Current exploration rate
        self.cer = None
        
        # Learning rate
        self.lr = lr
        
        # Initializer
        li = keras.initializers.RandomUniform(minval=-1, maxval=1, seed=None)
        
        # Define Keras model
        self.cm = Sequential()
        self.cm.add(Dense(9, input_dim=9, activation='sigmoid', kernel_initializer=li, use_bias=True))
        self.cm.add(Dense(6, activation='relu', kernel_initializer=li, use_bias=True))
        self.cm.add(Dense(1, activation='sigmoid', kernel_initializer=li, use_bias=False))
        
        # Compile model
        self.opt_cm = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, amsgrad=False)
        # Note acc = binary_accuracy (rounding of prediction)
        self.cm.compile(loss='mean_squared_error', optimizer=self.opt_cm)
        
        # Save state and reward for training NN
        self.statehist = np.zeros( (1000,9) ) * np.nan
        self.vhist = np.zeros(1000) * np.nan
        self.statecounter = 0
        
        # Game and reward history
        self.gh = []
        self.rh = []
    
    def initGame(self):      
        """ Start new game """
        # Exploration rate
        self.cer = np.exp(-self.policyupdates/self.eret)
        # State history of game
        self.sh = []
          
    def move(self, t, p=None):
        """
        Select move
        
        Parameters
        ----------
        t : TTT field
            Current board
        p : string
            policy
        """
        bmf = ttt.normalizeField(t.getFieldVector(), self.pl)
        
        hm, pms = t.hypoMoves(self.pl)
        hmn = ttt.normalizeField(hm, self.pl)

        vv = self.cm.predict_on_batch(hmn)
    
        if np.max(vv) == 0.0:
            # Choose random (loosing) move
            vv[::] = 1.0
        # Probability mass distribution for possible moves
        vvn = vv / np.sum(vv)
    
        if p == "greedy":
            # This must be a greedy move
            move = np.argmax(vvn)
        elif np.random.random() < self.cer:
            # Exploration phase, select move according to probability mass
            vvs = np.cumsum(vvn)
            x = np.random.random()
            move = np.where(x < vvs)[0][0]
        else:
            # This is a greedy move
            move = np.argmax(vvn)
 
        # Save (normalized) states before and after move       
        self.sh.append(bmf)
        self.sh.append(hmn[move,::])
         
        return pms[move]
 
    def update(self, r):
        """
        Game finished. Use reward, r, to update policy
        """

        self.policyupdates += 1
        
        # Save game in history
        self.gh.append(self.sh.copy())
        self.rh.append(r)
        
        ng = 100
        if len(self.gh) == ng:
            # Update network
            # Start with defined end states
            rs = np.array(self.rh)
            sh = np.zeros( (ng,9) )
            for i in range(ng):
                sh[i,::] = self.gh[i][-1]
            history_cm = self.cm.fit(sh, rs, \
                            epochs=10, batch_size=ng)
            
            for j in range(-2,-100,-1):
                # Go back through the games (current and next)
                cs = np.zeros( (ng,9) ) * np.nan
                ns = np.zeros( (ng,9) ) * np.nan
                for i in range(ng):
                    try:
                        cs[i,::] = self.gh[i][j]
                        ns[i,::] = self.gh[i][j+1]
                    except IndexError:
                        pass
                indi = (np.isnan(cs[::,0]) == False)
                if sum(indi) == 0:
                    break
                cs, ns = cs[indi,::], ns[indi,::]
                vnes = self.cm.predict_on_batch( ns )
                vnow = self.cm.predict_on_batch( cs )
                vnew = vnow + \
                    self.lr * (vnes - vnow)
                
                history_cm = self.cm.fit(cs, vnew, \
                                     epochs=4, batch_size=np.sum(indi))
            # Game and reward history
            self.gh = []
            self.rh = []
        
        

class QP(Player):
    """
    Q player
    
    Parameters
    ----------
    pl : int
        1 or 4 (crosses or circled)
    ri : boolean, optional
        If True, random initialization of Q
    dr : float
        Discount rate
    lr : float
        Learning rate
    eret : float
        Decay time of exploration rate
    """
    def __init__(self, pl, ri=True, name="Q_Player", eret=30000, lr=0.01):
        
        super().__init__(pl, name)
        
        self.policyupdates = 0
        self.eret = eret
        # Current exploration rate
        self.cer = None
        
        if ri:
            x = np.random.random(3**9)
        else:
            x = np.ones(3**9) * 0.5
        # Nine fields with three potential values 
        self.v = x.reshape( (3,3,3,3,3,3,3,3,3) )
        # Learning rate
        self.lr = lr
    
    def initGame(self):      
        """ Start new game """
        # Exploration rate
        self.cer = np.exp(-self.policyupdates/self.eret)
        # State history of game
        self.sh = []
          
    def move(self, t, p=None):
        """
        Select move
        
        Parameters
        ----------
        t : TTT field
            Current board
        p : string
            policy
        """
        # Before move field (normalized)
        bmf = ttt.normalizeField(t.getFieldVector(), self.pl)
        
        hm, pms = t.hypoMoves(self.pl)
        hmn = ttt.normalizeField(hm, self.pl)
        
        # Values of potential moves
        vv = np.zeros(len(pms))
        for i in range(len(pms)):
            vv[i] = self.v[tuple(hmn[i,::]+1)]
    
        if np.max(vv) == 0.0:
            # Choose random (loosing) move
            vv[::] = 1.0
        # Probability mass distribution for possible moves
        vvn = vv / np.sum(vv)
    
        if p == "greedy":
            # This must be a greedy move
            move = np.argmax(vvn)
        elif np.random.random() < self.cer:
            # Exploration phase, select move according to probability mass
            vvs = np.cumsum(vvn)
            x = np.random.random()
            move = np.where(x < vvs)[0][0]
        else:
            # This is a greedy move
            move = np.argmax(vvn)
 
        # Save (normalized) states before and after move       
        self.sh.append(bmf)
        self.sh.append(hmn[move,::])
         
        return pms[move]
    
    def update(self, r):
        """
        Game finished. Use reward, r, to update policy
        """
        self.policyupdates += 1

        # Index list for assessing value map      
        il = [tuple(s+1) for s in self.sh]
        
        # Value of last board state is clear
        self.v[il[-1]] = r
        # Update value for history
        for i in range(len(il)-2,-1,-1):
            # Value of potential upcoming moves
            self.v[il[i]] = self.v[il[i]] + self.lr * (self.v[il[i+1]] - self.v[il[i]])

    def saveVMap(self, name=None):
        """
        Parameters
        ----------
        name : string, optional
            Player name
        """
        if name is None:
            name = self.name
        name = name.replace(' ', '_')
        fn = name + "_vmap.npz"
        np.savez_compressed(fn, vmap=self.v)
    
    def loadVMap(self, name=None):
        """
        Parameters
        ----------
        name : string, optional
            Player name
        """
        if name is None:
            name = self.name
        name = name.replace(' ', '_')
        fn = name + "_vmap.npz"
        self.v = np.load(fn)["vmap"]
    

def tournament(pls, n=1000):
    """
    Play tournament of players (everybody against everybody as long as they play opposite same side)
    """
    
    print("Entering tournament with players:")
    for p in pls:
        print("    " + p.name + " playing " + {1:"x",4:"o"}[p.pl])
    print()
    
    for pair in itertools.combinations(pls, 2):
    
        if pair[0].pl == pair[1].pl:
            # Invalid pairing
            continue

        p1, p2 = sorted(pair, key=lambda x:x.pl)
    
        nx, no, nd = 0,0,0
        for i in range(n):
            
            t = ttt.TTT()
            p1.initGame()
            p2.initGame()
            
            while True:
                m = p1.move(t, p="greedy")
                w = t.move(m, p1.pl)
                if not w is None:
                    break            
                m = p2.move(t, p="greedy")
                w = t.move(m, p2.pl)
                if not w is None:
                    break    
            
            if w == 1:
                nx += 1
            elif w == -1:
                nd += 1
            else:
                no += 1
                
        print("Performance of " + p1.name + " (x) against " + p2.name + " (o)")
        print("    x wins %6.2f %%  and o wins %6.2f %%" % (nx/n*100, no/n*100) )
        print("    draws %6.2f %%" % (nd/n*100))
        print("    x looses %6.2f %% and o looses %6.2f %%" % ((n-nx-nd)/n*100, (n-no-nd)/n*100))
        print()


def train(pls, rounds):
    
    # Rewards for win, loose, and draw
    rw, rl, rd = 1., 0., 0.5
    
    print("Entering training with players:")
    for p in pls:
        print("    " + p.name + " playing " + {1:"x",4:"o"}[p.pl])
    print()
    
    for pair in itertools.combinations(pls, 2):
    
        if pair[0].pl == pair[1].pl:
            # Invalid pairing
            continue

        p1, p2 = sorted(pair, key=lambda x:x.pl)
        print("Training rounds: ", p1, " vs. ", p2)


        for j in range(rounds):
            # Training can begin
            
            if not silent:
                print("Game no. %5d" % j)
            
            t = ttt.TTT()
            p1.initGame()
            p2.initGame()
            
            # Make moves until victory
            while True:
                
                m = p1.move(t)
                w = t.move(m, p1.pl)
                if not w is None:
                    break
                
                m = p2.move(t)
                w = t.move(m, p2.pl)
                if not w is None:
                    break
    
            if not silent:
                t.printHistory()
            
            # Determine winner    
            if (w == 1):
                if not silent: print("Winner is 'x'")
                p1.update(rw)
                p2.update(rl)
            elif (w == 4):
                if not silent: print("Winner is 'o'")
                p1.update(rl)
                p2.update(rw)
            elif w == -1:
                if not silent: print("Draw")
                p1.update(rd)
                p2.update(rd)
            else:
                raise(ValueError("OOpps: " + str(w)))
         
            if not silent:
                print()
                print("-"*80)
                print()
           


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Tic Tac Toe')
    parser.add_argument('--n', type=int, default=10000,
                        help='Number of training rounds')
    parser.add_argument('--ntour', type=int, default=500,
                        help='Number of tournament rounds')
    
    parser.add_argument('--silent', action="store_true",
                        help='Stop output')
    
    args = parser.parse_args()
    
    # No. of training games
    n = args.n
    silent = args.silent
    
    
    # Set up two Q-Players for crosses and circles
    q1 = QP(1, name="qp_x", eret=30000, lr=0.05)
    q2 = QP(4, name="qp_o", eret=30000, lr=0.05)
    q1n = QP_nn(1, name="qpnn_x", eret=30000, lr=0.05)
    q2n = QP_nn(4, name="qpnn_o", eret=30000, lr=0.05)
    rp1 = RP(1, "rp_x")
    rp2 = RP(4, "rp_o")
    
    
    print("Tournament before training")
    tournament([q1,q2,q1n,q2n,rp1,rp2], args.ntour)
    train([q1,q2,q1n,q2n,rp1,rp2], n)
    print("Tournament after training")
    tournament([q1,q2,q1n,q2n,rp1,rp2], args.ntour)
