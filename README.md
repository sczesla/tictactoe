# Tic Tac Toe by reinforcement learning

TTT is a simple game with well known properties. The game is here used as a test case for a ''Reinforcement Learning'' algorithm based on the 'value function'. 

## Game statistics

TTT allows for about 255.000 different games to be played. Of these,
51% end with a win for the first player (x), 31% with win for the second player (o), and 18% lead to a draw. Dropping games which can be obtained by rotation and reflection from another game, the number of possible games reduces to 26.800. It can be shown that both players can force a draw. A perfect player, therefore, never looses a game.

### Random tournament
A random player is one who randomly selects one of the possible moves.
If two such players meet in a tournament, the first player (x)
wins about 59%, looses 29%, and draws 12% of the games. Equivalently, the second player (o) wins 29%, looses 59%, and draws 12% of the games.

## Learning the value function
The value function, V(S), assigns to every board state, S, the expected reward for the player starting in this state, given the current policy. In the case of TTT, the value function can be stored in a table with less than $3^9 \approx 20.000$ entries. In the course of training, an approximation of the optimal value function needs to be obtained. To that end, a series of training games is played. The agents start with a high but exponentially decreasing level of explorative behavior and an iterative scheme is applied to update the value function such that

<img src="https://render.githubusercontent.com/render/math?math=V(S_i) = V(S_i) + \alpha(V(S_{i+1}) - V(S_i) ) ">

(cf. monograph by Sutton and Barto). The value of the final state is taken to be identical to the received reward with numerical values of one for a win, 0.5 for a draw and zero for a loss.

## Implementation of players

The value function and the iterative scheme can be implemented in many flavors. I here use both a table to store the value function (QP) and an artificial neural network (ANN) implemented using Keras and Tensorflow (QP_nn).  

## Results of learning
Any agent having received successful training should outperform a random player. As any player can force a draw, a properly trained agent should not loose a game. Similar results can be reproduced by calling

    python tttrain.py --silent > log.log

### Performance against random opponent before training
The following table gives the percentages of wins, draws, and losses before training for x and o players with table representation of value function (qp) and ANN representation (qpnn). 

| Player  | Win   | Draw   | Loss |
|---------|-------|--------|-------|
| qp_x    | 69.2  | 9.8    | 21    |
| qp_o    | 30.4  | 9.6    | 60    |
| qpnn_x  | 41.2  | 5.8    | 53    |
| qpnn_o  | 43.8  | 9.2    | 47    |

### Performance against random opponent after training

After training, the performance of the players improved substantially.

| Player  | Win   | Draw   | Loss |
|---------|-------|--------|------|
| qp_x    | 99.2  | 0.8    | 0    |
| qp_o    | 91.0  | 8.6    | 0.4  |
| qpnn_x  | 92.0  | 8.0    | 0    |
| qpnn_o  | 85.2  | 12.0   | 2.8  |

The worst performance is exhibited by qpnn_o, i.e., the second player (o) whose value function is approximated by an ANN. Further training improves its performance. 
