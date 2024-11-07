from GLOBAL_VARs import *
import copy


class GeneticAlgoAgent:
    """
    A class to represent an agent in the environment.

    Attributes:
        y (int): The agent's y-coordinate.
        x (int): The agent's x-coordinate.
        turn (bool): A flag to track whether it's the agent's turn to move.
    """

    def __init__(self, y, x):
        """
        Initialize the agent's starting position.

        Args:
            y (int): The agent's initial y-coordinate.
            x (int): The agent's initial x-coordinate.
        """
        # Set up the agent's starting position
        self.y = y
        self.x = x
        self.scores = 0
        self.state = 0
        self.dead = False

        # Set up the turn flag
        self.turn = True

        # Set up memory and actions
        self.memory_size = 30
        self.n_mutations = 3
        self.actions = np.random.randint(0, len(ACTION), self.memory_size)
        self.memory = self.actions

        # Store the initial position
        self.oy = y
        self.ox = x

    def copy(self):
        """
        Create a deep copy of the agent.
        """
        return copy.deepcopy(self)

    def take_action(self):
        """
        Perform an action based on the agent's memory.
        """
        if len(self.memory)!= 0 and not self.dead:
            self.move(self.memory[0])
            try:
                self.memory = self.memory[1:]
            except:
                pass
        else:
            self.dead = True

    def mutate(self):
        """
        Mutate the agent's actions.
        """
        rand_int = np.random.randint(0, len(self.actions), self.n_mutations)
        self.actions[rand_int] = np.random.randint(0, len(ACTION))

    def reset_turn(self):
        """
        Reset the agent's turn flag to True.
        """
        self.turn = True

    def end_turn(self):
        """
        Set the agent's turn flag to False.
        """
        self.turn = False

    def move(self, direction):
        """
        Move the agent in a specific direction.

        Args:
            direction (str): The direction to move (UP, DOWN, LEFT, RIGHT).
        """
        if direction == ACTION['UP']:
            return self.UP()
        elif direction == ACTION["DOWN"]:
            return self.DOWN()
        elif direction == ACTION["LEFT"]:
            return self.LEFT()
        elif direction == ACTION["RIGHT"]:
            return self.RIGHT()

    def move_to(self, y, x):
        """
        Move the agent to a new position.

        Args:
            y (int): The new y-coordinate.
            x (int): The new x-coordinate.

        Returns:
            bool: Whether the move was successful.
        """
        # Ensure the agent stays within the grid
        x = max(0, min(x, GRID_WIDTH - 1))
        y = max(0, min(y, GRID_HEIGHT - 1))

        if not self.turn:
            # Do not move if it's not the agent's turn
            return False

        if core_mechanics.collision_detection(self, y, x):
            # Handle collision detection
            pass
        else:
            # Update the agent's position
            self.y = y
            self.x = x

        # End the agent's turn
        self.end_turn()
        return True

    def UP(self):
        """
        Move the agent up.
        """
        if self.x is None:
            return
        y = self.y - 1
        x = self.x
        return self.move_to(y, x)

    def DOWN(self):
        """
        Move the agent down.
        """
        if self.x is None:
            return
        y = self.y + 1
        x = self.x
        return self.move_to(y, x)

    def RIGHT(self):
        """
        Move the agent right.
        """
        if self.x is None:
            return
        y = self.y
        x = self.x + 1
        return self.move_to(y, x)

    def LEFT(self):
        """
        Move the agent left.
        """
        if self.x is None:
            return
        y = self.y
        x = self.x - 1
        return self.move_to(y, x)
    
    
class QLEARNING_ALGO_AGENT:
    """
    A class to represent an agent in the environment.

    Attributes:
        y (int): The agent's y-coordinate.
        x (int): The agent's x-coordinate.
        turn (bool): A flag to track whether it's the agent's turn to move.
    """

    def __init__(self, y, x) -> None:
        """
        Initialize the agent's starting position.

        Args:
            y (int): The agent's initial y-coordinate.
            x (int): The agent's initial x-coordinate.
        """
        # Set up the agent's starting position
        self.y = y
        self.x = x
        
        self.scores = 0
        self.state = 0
        self.dead = False
        
        # Set up the turn flag
        self.turn = True
        
        #Q_TABLE PROPERTIES:
        self.lookup_table = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(ACTION)))
        # the look-up table consited of 2 state: y, x and 4 action
        self.ALPHA = 0.3
        self.GAMMA = 0.9
        self.EPS = 0
        self.number_of_action = 20
        self.number_of_action_remain = self.number_of_action
        
        #original pos
        self.oy = y
        self.ox = x
        
    def copy(self):
        return copy.deepcopy(self)
    
#----------------------------------------
    def update_table(self, state: list, action: int, next_state: list, next_action: int, reward):
        self.lookup_table[tuple(state + [action])] += self.ALPHA * (reward + self.GAMMA * self.lookup_table[tuple(next_state + [next_action])]- self.lookup_table[tuple(state + [action])])
        
    
    def max_action(self, state: list):
        action = np.argmax(self.lookup_table[tuple(state)])
        
        return action
    
    def sample_action(self, state: list) -> int:
        action = np.random.randint(0, len(ACTION))
        
        return action
    
    def take_step(self, want_next_state=False) -> tuple[list[int], int, list[int], int]:
        if self.number_of_action_remain > 0 and self.dead != True:
            rng = np.random.rand()
            cur_state = [self.y, self.x]
            if rng > self.EPS and want_next_state == False:
                action = self.sample_action(cur_state)
            else:
                action = self.max_action(cur_state)
            self.move(action)
            next_state = [self.y, self.x]
            reward = REWARD[STATES[self.state]] 
            
            self.number_of_action_remain -= 1
            return cur_state, action, next_state, reward
        else:
            self.dead = True
         
#----------------------------------------
    def take_action(self) -> int:
        try:
            self.EPS += 0.001
            cur_state, action, next_state, reward = self.take_step()
            next_state, next_action, far_state, _ = self.take_step(want_next_state=True)
            # print(f"cur_state: {cur_state}, action: {action}, next_state: {next_state}, next_action: {next_action}, far_state: {far_state}, reward: {reward}")
            print(self.lookup_table)
            self.update_table(cur_state, action, next_state, next_action, reward)
        except TypeError:
            pass
        
    def reset_turn(self):
        """
        Reset the agent's turn flag to True.
        """
        self.turn = True

    def end_turn(self):
        """
        Set the agent's turn flag to False.
        """
        self.turn = False
    
    def move(self, direction: str):
        if direction == ACTION['UP']:
            return self.UP()
        elif direction == ACTION["DOWN"]:
            return self.DOWN()
        elif direction == ACTION["LEFT"]:
            return self.LEFT()
        elif direction == ACTION["RIGHT"]:
            return self.RIGHT()
            
    def move_to(self, y, x):
        """
        Move the agent to a new position.

        Args:
            y (int): The new y-coordinate.
            x (int): The new x-coordinate.

        Returns:
            None
        """
        # Ensure the agent stays within the grid
        x = max(0, min(x, GRID_WIDTH - 1))
        y = max(0, min(y, GRID_HEIGHT - 1))
        if self.turn == False:
            # Do not move if it's not the agent's turn
            return None

        if core_mechanics.collision_detection(self, y, x):
            # Handle collision detection
            pass
        else:
            # Update the agent's position
            self.y = y
            self.x = x

        # End the agent's turn
        self.end_turn()
        # movement_clock.tick(MOVEMENT_FPS)
        return True

    def UP(self):
        """
        Move the agent up.
        """
        if self.x == None:
            return
        y = self.y - 1
        x = self.x
        return self.move_to(y, x)

    def DOWN(self):
        """
        Move the agent down.
        """
        if self.x == None:
            return
        y = self.y + 1
        x = self.x
        return self.move_to(y, x)

    def RIGHT(self):
        """
        Move the agent right.
        """
        if self.x == None:
            return
        y = self.y
        x = self.x + 1
        return self.move_to(y, x)

    def LEFT(self):
        """
        Move the agent left.
        """
        if self.x == None:
            return
        y = self.y
        x = self.x - 1
        return self.move_to(y, x)
    
    
    
class Deep_Q_LEARNING_ALGO_AGENT:
    """
    A class to represent an agent in the environment.

    Attributes:
        y (int): The agent's y-coordinate.
        x (int): The agent's x-coordinate.
        turn (bool): A flag to track whether it's the agent's turn to move.
    """

    def __init__(self, y, x) -> None:
        """
        Initialize the agent's starting position.

        Args:
            y (int): The agent's initial y-coordinate.
            x (int): The agent's initial x-coordinate.
        """
        # Set up the agent's starting position
        self.y = y
        self.x = x
        
        self.scores = 0
        self.state = 0
        self.dead = False

        # Set up the turn flag
        self.turn = True
        
        #Q_TABLE PROPERTIES:
        self.memory = []
        self.lookup_table = np.zeros((GRID_HEIGHT, GRID_WIDTH, len(ACTION)))
        
        # the look-up table consited of 2 state: y, x and 4 action
        self.ALPHA = 0.3
        self.GAMMA = 0.9
        self.EPS = 0
        self.number_of_action = 20
        self.number_of_action_remain = self.number_of_action
        
        #original pos
        self.oy = y
        self.ox = x
        
#     def copy(self):
#         return copy.deepcopy(self)
    
# #----------------------------------------
#     def update_table(self, state: list, action: int, next_state: list, next_action: int, reward):
#         self.lookup_table[tuple(state + [action])] += self.ALPHA * (reward + self.GAMMA * self.lookup_table[tuple(next_state + [next_action])]- self.lookup_table[tuple(state + [action])])
        
    
#     def max_action(self, state: list):
#         action = np.argmax(self.lookup_table[tuple(state)])
        
#         return action
    
#     def sample_action(self, state: list) -> int:
#         action = np.random.randint(0, len(ACTION))
        
#         return action
    
#     def take_step(self, want_next_state=False) -> tuple[list[int], int, list[int], int]:
#         if self.number_of_action_remain > 0 and self.dead != True:
#             rng = np.random.rand()
#             cur_state = [self.y, self.x]
#             if rng > self.EPS and want_next_state == False:
#                 action = self.sample_action(cur_state)
#             else:
#                 action = self.max_action(cur_state)
#             self.move(action)
#             next_state = [self.y, self.x]
#             reward = REWARD[STATES[self.state]] 
            
#             self.number_of_action_remain -= 1
#             return cur_state, action, next_state, reward
#         else:
#             self.dead = True
         
# #----------------------------------------
#     def take_action(self) -> int:
#         try:
#             self.EPS += 0.001
#             cur_state, action, next_state, reward = self.take_step()
#             next_state, next_action, far_state, _ = self.take_step(want_next_state=True)
#             # print(f"cur_state: {cur_state}, action: {action}, next_state: {next_state}, next_action: {next_action}, far_state: {far_state}, reward: {reward}")
#             print(self.lookup_table)
#             self.update_table(cur_state, action, next_state, next_action, reward)
#         except TypeError:
#             pass
        
#     def reset_turn(self):
#         """
#         Reset the agent's turn flag to True.
#         """
#         self.turn = True

#     def end_turn(self):
#         """
#         Set the agent's turn flag to False.
#         """
#         self.turn = False
    
#     def move(self, direction: str):
#         if direction == ACTION['UP']:
#             return self.UP()
#         elif direction == ACTION["DOWN"]:
#             return self.DOWN()
#         elif direction == ACTION["LEFT"]:
#             return self.LEFT()
#         elif direction == ACTION["RIGHT"]:
#             return self.RIGHT()
            
#     def move_to(self, y, x):
#         """
#         Move the agent to a new position.

#         Args:
#             y (int): The new y-coordinate.
#             x (int): The new x-coordinate.

#         Returns:
#             None
#         """
#         # Ensure the agent stays within the grid
#         x = max(0, min(x, GRID_WIDTH - 1))
#         y = max(0, min(y, GRID_HEIGHT - 1))
#         if self.turn == False:
#             # Do not move if it's not the agent's turn
#             return None

#         if core_mechanics.collision_detection(self, y, x):
#             # Handle collision detection
#             pass
#         else:
#             # Update the agent's position
#             self.y = y
#             self.x = x

#         # End the agent's turn
#         self.end_turn()
#         # movement_clock.tick(MOVEMENT_FPS)
#         return True

#     def UP(self):
#         """
#         Move the agent up.
#         """
#         if self.x == None:
#             return
#         y = self.y - 1
#         x = self.x
#         return self.move_to(y, x)

#     def DOWN(self):
#         """
#         Move the agent down.
#         """
#         if self.x == None:
#             return
#         y = self.y + 1
#         x = self.x
#         return self.move_to(y, x)

#     def RIGHT(self):
#         """
#         Move the agent right.
#         """
#         if self.x == None:
#             return
#         y = self.y
#         x = self.x + 1
#         return self.move_to(y, x)

#     def LEFT(self):
#         """
#         Move the agent left.
#         """
#         if self.x == None:
#             return
#         y = self.y
#         x = self.x - 1
#         return self.move_to(y, x)