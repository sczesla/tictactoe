import numpy as np

class TTT:
    
    def getIndicesforLine(self, n):
        """
        Return indices [(x,y), ...] for line no. n
        
        0,1,2 : horizontal
        3,4,5 : vertical
        6 : left-right diagonal
        7 : right-left diagonal
        """
        if 0 <= n < 3:
            # horizontal
            return [(0,n), (1,n), (2,n)]
        elif 3 <= n < 6:
            # vertical
            return [(n-3,0), (n-3,1), (n-3,2)]
        elif n == 6:
            # first diagonal
            return [(0,0), (1,1), (2,2)]
        elif n == 7:
            # second diagonal
            return [(0,2), (1,1), (0,0)]
        else:
            raise(ValueError("Line no. " + str(n) + " does not exist. Choose 0-7."))
    
    def getLineSums(self):
        """
        Return list with sums along lines 0-7
        """
        ls = np.zeros(8, dtype=int)
        for i in range(3):
            ls[i] = self.field[::,i].sum()
            ls[i+3] = self.field[i,::].sum()
            ls[6] += self.field[i,i]
            ls[7] += self.field[2-i,i]
        return ls
    
    def isWon(self):
        """
        Returns 1 or 4 if "x" or "o" won. -1 for a draw. None otherwise
        """
        ls = self.getLineSums()
        if 3 in ls:
            return 1
        elif 12 in ls:
            return 4
        elif not (0 in self.field):
                return -1
        else:
            return None
    
    def printField(self):
        for y in range(3):
            print("|".join([self.encoding[self.field[x,y]] for x in range(3)]))
            if y < 2: print("-"*5)
    
    def printHistory(self):
        lines = [""]*5
        for y in range(3):
            for z in range(len(self.history)):
                lines[y*2] += " . " + "|".join([self.encoding[self.history[z][2][x,y]] for x in range(3)])
                if y < 2: lines[y*2+1] += " . " + ("-"*5)        
        print("Game history")
        for i in range(5):
            print(lines[i])
            
    
    def validMoves(self):
        """
        Returns the numbers (0-8) of empty fields
        """
        indi = np.where(self.field == 0)
        return self._xy_to_n(indi[0], indi[1])
        
    def _n_to_xy(self, n):
        """
        Convert field no. (0-8) into x,y field coordinates
        """
        y, x = divmod(n, 3)
        return x, y
    
    def _xy_to_n(self, x, y):
        """
        Return field coordinated x, y into field number (0-8)
        """
        return x + y*3
    
    def getFieldVector(self):
        """
        Get the field as a 1d array
        """
        return self.field.T.flatten()
    
    def hypoMoves(self, m):
        """
        All valid moves with field set to m
        
        Returns
        -------
        fields : 2d array
            An array with shape (no. of valid moves, 9) so that a[0,::] gives
            the field as it would look after the first valid move for player
            indicated by 'm'.
        pm : array
            Possible moves (number of field)
        """
        pm = self.validMoves()
        ff = self.getFieldVector().repeat(len(pm)).reshape( (9, len(pm)) ).T
        # Now e.g. ff[n,::] contains the n-th copy of the field 
        ff[np.arange(len(pm)), pm] = m
        return ff, pm
    
    def move(self, n, m):
        """
        Set field no. n to m and returns output of isWon
        """
        x, y = self._n_to_xy(n)
        if self.field[x,y] != 0:
            raise(ValueError("Invalid move"))
        if not m in [1,4]:
            raise(ValueError("Invalid move"))
        self.field[x,y] = m
        self.history.append((n, m, self.field.copy()))
        return self.isWon()
    
    def __init__(self):
        self.encoding = {0:" ", 1:"x", 4:"o"}
        
        self.field = np.zeros((3,3), dtype=int)
        
        # Game history (list of tuples of field no. and player ID)
        self.history = []
        
        self.allLines = []
        for n in range(8):
            self.allLines.append(self.getIndicesforLine(n))
       

        
def normalizeField(f, ap):
    """
    Return normalized field (active player=1, opponent=-1, empty=0)
    """
    f = f.copy()
    if ap == 1:
        f[f > 3] -= 5
    elif ap == 4:
        f[f == 1] -= 2
        f[f == 4] = 1
    else:
        raise("Invalid active player (ap)")    
    return f
            
