import pygame
import sys

from GLOBAL_VARs import *
from agents import *
from utils import export_numpy_array, import_numpy_array

# Initialize the editing mode display
pygame.display.set_caption(f"{global_instances.Editing_Mode=}")

def handle_events():
    """
    Handle user input events, such as mouse clicks and key presses.
    """
    for event in pygame.event.get():
        # Quit the game if the user closes the window
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Handle mouse clicks
        elif event.type == pygame.MOUSEBUTTONDOWN:
            handle_mouse_click(event)

        # Handle key presses
        elif event.type == pygame.KEYDOWN:
            handle_key_press(event)

def handle_mouse_click(event):
    """
    Handle mouse clicks by updating the grid based on the current editing mode.
    """
    x = event.pos[0] // CELL_SIZE
    y = event.pos[1] // CELL_SIZE

    if global_instances.Editing_Mode == "obstacle":
        # Mark the cell as an obstacle
        world_rendering.grid[y, x] = 1
    elif global_instances.Editing_Mode == "terminal":
        # Mark the cell as a terminal
        world_rendering.grid[y, x] = 2
    elif global_instances.Editing_Mode == "place agent":
        # Mark the cell as the agent's starting position
        # agent = GeneticAlgoAgent(y, x)
        agent = QLEARNING_ALGO_AGENT(y, x)
        agent_population.add_pop(agent)
        
    elif global_instances.Editing_Mode == "erase":
        # Clear the cell
        world_rendering.grid[y, x] = 0
        agent_population.pops = [pop for pop in agent_population.pops if (pop.y != y) or (pop.x != x)]

def handle_key_press(event):
    """
    Handle key presses by updating the editing mode.
    """
    if event.key == pygame.K_o:
        # Switch to obstacle mode
        global_instances.Editing_Mode = "obstacle"
    elif event.key == pygame.K_t:
        # Switch to terminal mode
        global_instances.Editing_Mode = "terminal"
    elif event.key == pygame.K_s:
        # Switch to save mode
        global_instances.Editing_Mode = "save"
    elif event.key == pygame.K_i:
        # Switch to import mode
        global_instances.Editing_Mode = "import"
    elif event.key == pygame.K_a:
        # Switch to place agent mode
        global_instances.Editing_Mode = "place agent"
    elif event.key == pygame.K_BACKSPACE:
        # Switch to erase mode
        global_instances.Editing_Mode = "erase"

def handle_editing_mode():
    """
    Save or import the grid if the corresponding editing mode is active.
    """
    if global_instances.Editing_Mode == "save":
        # Save the grid to a file
        export_numpy_array(world_rendering.grid, ENV_DATA_PATH)
        agent_population.export_pops()
    elif global_instances.Editing_Mode == "import":
        # Import the grid from a file
        world_rendering.grid = import_numpy_array(ENV_DATA_PATH)
        agent_population.import_pops()
        
        
# Main game loop
while True:
    # Handle user input events
    handle_events()

    # Update the editing mode display
    pygame.display.set_caption(f"{global_instances.Editing_Mode=}")

    # Render the world
    world_rendering.world_render(agent_population)

    # Update the display
    pygame.display.flip()

    # Save or import the grid if the corresponding editing mode is active
    handle_editing_mode()

    # Cap the frame rate to 60 FPS
    pygame.time.delay(FPS)