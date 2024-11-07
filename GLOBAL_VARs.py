# Import necessary libraries
import pygame

from utils import *
import itertools
import operator

# Initialize Pygame
pygame.init()
pygame.mixer.init()

terminated_sound = pygame.mixer.Sound('./ES_Impact Metal 1 - SFX Producer.mp3')

# Set up the clock
clock = pygame.time.Clock()
FPS = 60
# Set up the movement clock
movement_clock = pygame.time.Clock()
MOVEMENT_FPS = 60

# Set up some constants for the game window and grid
WIDTH = 800  # The width of the game window
HEIGHT = 600  # The height of the game window
CELL_SIZE = 100  # The size of each cell in the grid
GRID_WIDTH = WIDTH // CELL_SIZE  # The number of cells in the grid horizontally
GRID_HEIGHT = HEIGHT // CELL_SIZE  # The number of cells in the grid vertically

# Set up some colors
WHITE = (255, 255, 255)  # White color
BLACK = (0, 0, 0)  # Black color
RED = (255, 0, 0)  # Red color
GREEN = (0, 255, 0)  # Green color
BLUE = (0, 0, 255)  # Blue color

# Set up the display
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT))  # Create a game window with the specified width and height
# Set up the hover panel
HOVER_PANEL = pygame.Surface((WIDTH//6, HEIGHT//6))
HOVER_PANEL.fill(WHITE)  # White background
# Draw a black border around the surface
pygame.draw.rect(HOVER_PANEL, (0, 0, 0), (0, 0, HOVER_PANEL.get_width(), HOVER_PANEL.get_height()), 1)
# Create a Pygame surface to blit the plot onto
PLOT_SUR = pygame.Surface((WIDTH, HEIGHT))
		
# Set up the font
FONT = pygame.font.Font(None, 24)  # Create a font object with the default font and size 36
mouse_hover_FONT = pygame.font.SysFont("Arial", 12)


# Set up the path to the environment data file
ENV_DATA_PATH = './env_data/env_test_data.npy'

# Set up the editing mode
Editing_Mode = "start"  # The current editing mode

STR_STATE = {'agent': -1,
             'blank': 0,
             'obstacle': 1,
             'terminate': 2}

STATES = {-1: 'agent',
          0: 'blank',
          1: 'obstacle',
          2:'terminate'}

ACTION = {'UP': 0,
          'DOWN': 1,
          'LEFT': 2,
          'RIGHT': 3}

REWARD = {'agent': -1,
          'blank': -1,
          'obstacle': -1,
          'terminate': 10}

        

class CoreMechanics:
    """
    A class to handle core mechanics of the game.

    Attributes:
        None
    """

    def __init__(self) -> None:
        """
        Initialize the core mechanics.
        """
        pass

    def collide_with_obstacle(self, agent):
        """
        Check if the agent collides with an obstacle.

        Returns:
            bool: True if the agent collides with an obstacle, False otherwise.
        """
        agent.state = STR_STATE['obstacle']
        agent.scores += REWARD[STATES[STR_STATE['obstacle']]]
        return True

    def collide_with_terminal(self, agent):
        """
        Check if the agent collides with a terminal and delete it.

        Returns:
            bool: True if the agent collides with a terminal, False otherwise.
        """
        agent.state = STR_STATE['terminate']
        agent.scores += REWARD[STATES[STR_STATE['terminate']]]
        # agent.dead = True
        terminated_sound.play()
        return False

    def collide_with_agent(self, agent):
        """
        Check if the agent collides with another agent.

        Returns:
            bool: True if the agent collides with another agent, False otherwise.
        """
        agent.state = STR_STATE['agent']
        agent.scores += REWARD[STATES[STR_STATE['agent']]]
        return False

    def collision_detection(self, agent, y, x):
        """
        Detect collisions with obstacles, terminals, or agents.

        Args:
            y (int): The y-coordinate of the agent.
            x (int): The x-coordinate of the agent.

        Returns:
            bool: True if a collision is detected, False otherwise.
        """
        if world_rendering.grid[y, x] == 1:
            # Check for collision with obstacle
            return self.collide_with_obstacle(agent)
        elif world_rendering.grid[y, x] == 2:
            # Check for collision with terminal
            # world_rendering.grid[y, x] = 0
            return self.collide_with_terminal(agent)
        # elif world_rendering.grid[y, x] == -1:
        #     # Check for collision with agent
        #     self.collide_with_agent(agent)
        else:
            # No collision detected
            agent.state = STR_STATE['blank']
            agent.scores += REWARD[STATES[STR_STATE['blank']]]
            return False
        

class Population:
    """
    A class to manage a population of agents.

    Attributes:
        pops (list): A list of agents in the population.
    """

    def __init__(self) -> None:
        """
        Initialize the population.
        """
        self.pops = []
        self.max_score = []
        self.avg_score = []

    def __call__(self):
        """
        Return the population.
        """
        return(self.pops)

    def __getitem__(self, idx):
        """
        Get an agent from the population by index.

        Args:
            idx (int): The index of the agent.

        Returns:
            agent: The agent at the specified index.
        """
        return self.pops[idx]
    
    def genetic_algo_update(self):
        n_pop = len(self.pops)
        sorted_pop_by_scores = sorted(self.pops, key=lambda pop: (pop.scores), reverse=True)
        sorted_pop_by_dead = [pop for pop in self.pops if pop.dead == False]
        if len(sorted_pop_by_dead) == 0:
            self.max_score.append(sorted_pop_by_scores[0].scores)
            self.avg_score.append(np.mean([pop.scores for pop in sorted_pop_by_scores]))
            
            self.clear_all_pops()
            
            for i in range(n_pop):
                new_pop = sorted_pop_by_scores[0].copy()
                
                new_pop.mutate()
                new_pop.memory = new_pop.actions
                new_pop.y = new_pop.oy
                new_pop.x = new_pop.ox
                new_pop.scores = 0
                new_pop.dead = False
                
                self.add_pop(new_pop)
                
    def Q_learning_update(self):
        n_pop = len(self.pops)
        sorted_pop_by_scores = sorted(self.pops, key=lambda pop: (pop.scores), reverse=True)
        sorted_pop_by_dead = [pop for pop in self.pops if pop.dead == False]
        if len(sorted_pop_by_dead) == 0:
            self.max_score.append(sorted_pop_by_scores[0].scores)
            self.avg_score.append(np.mean([pop.scores for pop in sorted_pop_by_scores]))
            
            self.clear_all_pops()
            
            for i in range(n_pop):
                new_pop = sorted_pop_by_scores[i].copy()
                
                new_pop.y = new_pop.oy
                new_pop.x = new_pop.ox
                new_pop.scores = 0
                new_pop.dead = False
                new_pop.number_of_action_remain = new_pop.number_of_action
                
                self.add_pop(new_pop)
                        
    def take_action(self):
        for pop in self.pops:
            pop.take_action()
    
    def sort_stack_pops(self):
        removed_NONE = [pop for pop in self.pops if pop.x != None]
        sorted_pop = sorted(removed_NONE, key=lambda pop: (pop.y, pop.x))
        instance_groups = itertools.groupby(sorted_pop, key=operator.attrgetter('y', 'x'))
        stacked_pops = [tuple(group) for key, group in instance_groups]

        return stacked_pops
    
    def add_pop(self, pop):
        """
        Add an agent to the population.

        Args:
            pop (agent): The agent to add.
        """
        self.pops.append(pop)

    def import_pops(self, name = 'Agents'):
        self.pops = pickle_import(name=name)
    
    def export_pops(self, name = 'Agents'):
        pickle_export(self.pops, name=name)
        
    def clear_all_pops(self):
        """
        Clear all agents from pops
        """
        self.pops = []
        
    def reset_turn(self):
        """
        Reset the turn of all agents in the population.
        """
        [pop.reset_turn() for pop in self.pops]
        
        

        
class WorldRendering:
    """
    A class to handle world rendering.
    """
    def __init__(self) -> None:
        """
        Initialize the world rendering.

        Attributes:
            grid (np.ndarray): A 2D grid with zeros
        """
        # Set up the grid
        self.grid = np.zeros((GRID_HEIGHT, GRID_WIDTH))  # Create a 2D grid with zeros

    def world_render(self, population: Population) -> None:
        """
        Render the world.

        This method draws the grid and fills in the cells with different colors based on their values.
        """
        y_obstacle, x_obstacle = list(np.where(self.grid == STR_STATE['obstacle']))
        y_terminal, x_terminal = list(np.where(self.grid == STR_STATE['terminate']))
        # Draw the grid
        SCREEN.fill(WHITE)  # Fill the screen with white color
        
        self.render_obstacles(y_obstacle, x_obstacle)
        self.render_terminals(y_terminal, x_terminal)
        self.render_pops(population)
        
        for y in range(GRID_HEIGHT):
            for x in range(GRID_WIDTH):
                # Draw a rectangle for each cell
                RECT = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(SCREEN, BLACK, RECT, 1)  # Draw a rectangle with a line width of 2
                #DRAW LINES
                # pygame.draw.line(SCREEN, BLACK, (x* CELL_SIZE, y* CELL_SIZE), (x*CELL_SIZE+CELL_SIZE, y*CELL_SIZE+CELL_SIZE), 1)
                # pygame.draw.line(SCREEN, BLACK, (x*CELL_SIZE+CELL_SIZE, y*CELL_SIZE), (x*CELL_SIZE, y*CELL_SIZE+CELL_SIZE), 1)
                
                    
    def render_obstacles(self, y_obstacle, x_obstacle):
        for y, x in zip(y_obstacle, x_obstacle):
            pygame.draw.rect(SCREEN, RED, (x * CELL_SIZE + CELL_SIZE // 4, y * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2))
            
    def render_terminals(self, y_terminal, x_terminal):
        for y, x in zip(y_terminal, x_terminal):
            pygame.draw.rect(SCREEN, GREEN, (x * CELL_SIZE + CELL_SIZE // 4, y * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2))
                   
    def render_pops(self, population: Population):
        stacked_pops = population.sort_stack_pops()
        for stack in stacked_pops:
            text = FONT.render(str(len(stack)), True, RED)
            SCREEN.blit(text, (stack[0].x * CELL_SIZE + 0.7*CELL_SIZE, stack[0].y * CELL_SIZE))
            for pop in stack:
                if pop.dead != True:
                    rect = pygame.Rect((pop.x * CELL_SIZE + CELL_SIZE // 4, pop.y * CELL_SIZE + CELL_SIZE // 4, CELL_SIZE // 2, CELL_SIZE // 2))
                    
                    pygame.draw.rect(SCREEN, BLACK, rect)
        
# Create instances of the classes
class GlobalInstances:
    """
    A class to hold global instances.
    """
    def __init__(self) -> None:
        """
        Initialize the global instances.

        Attributes:
            Editing_Mode (str): The current editing mode
        """
        self.Editing_Mode ='start'  # The current editing mode


    
global_instances = GlobalInstances()
core_mechanics = CoreMechanics()
world_rendering = WorldRendering()

agent_population = Population()