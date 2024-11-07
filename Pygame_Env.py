import pygame
import sys

from GLOBAL_VARs import *
from agents import *
from utils import import_numpy_array, pickle_export, pickle_import

# Set up the pause variable
paused = False  # Flag to toggle the game's pause state
mouse_hover = False
plot = False
plot_image = None

# Game loop
def main_game_loop():
    """
    The main game loop that runs indefinitely until the user quits.
    """
    world_rendering.grid = import_numpy_array(ENV_DATA_PATH)  # Load environment data from file
    agent_population.import_pops()  # Load the agent population
    clock = pygame.time.Clock()  # Initialize the game clock

    while True:
        # Event loop
        handle_events()

        # Update game logic
        if not paused:
            update_game_logic()

        # Update display
        update_display()

        # Cap the frame rate
        clock.tick(FPS)
        pygame.display.set_caption(f"{clock.get_fps()=:.2f}\t{movement_clock.get_fps()=:.2f}\
        \t{paused=}\t\t{global_instances.Editing_Mode}")
def handle_events():
    """
    Handle user input events, such as key presses and quit events.
    """
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            handle_key_press(event.key)
            
def handle_mouse_hover():
    """
    Handle mouse hovers by updating the grid based on the current editing mode.
    """
    if mouse_hover:
        mouse_x = pygame.mouse.get_pos()[0] 
        mouse_y = pygame.mouse.get_pos()[1] 
        
        HOVER_PANEL.fill(WHITE)

        pos_pops = [pop for pop in agent_population() if (pop.x == mouse_x // CELL_SIZE) and (pop.y == mouse_y // CELL_SIZE)]
        
        data = [[pop.state, pop.scores] for pop in pos_pops]
        x = 10
        y = 10
        for row in data:
            row_str = ", ".join(map(str, row))  # Convert the row to a string
            text_surface = mouse_hover_FONT.render(row_str, True, BLACK)
            HOVER_PANEL.blit(text_surface, (x, y))
            y += text_surface.get_height() + 10  # Move down to the next line

        hover_panel_rect = HOVER_PANEL.get_rect()
        hover_panel_rect.centerx = mouse_x + 20
        hover_panel_rect.centery = mouse_y + 20
        
        # Draw a black border around the surface
        pygame.draw.rect(HOVER_PANEL, (0, 0, 0), (0, 0, HOVER_PANEL.get_width(), HOVER_PANEL.get_height()), 1)
        
        SCREEN.blit(HOVER_PANEL, hover_panel_rect)
        
def create_graph():
    # Create a matplotlib figure and axis
    fig, ax = plt.subplots()
    
    n = len(agent_population.max_score)
    x1 = np.linspace(0, n, n)
    x2 = np.linspace(0, n, n)
    
    y1 = agent_population.max_score
    y2 = agent_population.avg_score
    # Plot some data
    ax.plot(x1, y1)
    ax.plot(x2, y2)
    
    # Save the figure to a file
    plt.savefig('plots/plot.png', bbox_inches='tight')
    
    print(agent_population()[0].lookup_table, '\n')
    
    
def plot_graph():
    if plot:
        # Load the global image 
        global plot_image
        
        # Blit the image onto the surface
        PLOT_SUR.blit(plot_image, (0, 0))

        
        SCREEN.blit(PLOT_SUR, (0,0))

def handle_key_press(key):
    """
    Handle key press events, such as toggling the pause state or importing data.
    """
    if key == pygame.K_SPACE:
        toggle_pause()
    elif key == pygame.K_i:
        import_data()
    elif key == pygame.K_h:
        global mouse_hover
        mouse_hover = not mouse_hover
    elif key == pygame.K_p:
        create_graph()
        global plot
        global plot_image
        plot = not plot
        plot_image = pygame.image.load('plots/plot.png')
        
        

def toggle_pause():
    """
    Toggle the game's pause state.
    """
    global paused
    paused = not paused

def import_data():
    """
    Import environment data and the agent population from file.
    """
    global_instances.Editing_Mode = "import"
    # movement_clock.tick(MOVEMENT_FPS)
    world_rendering.grid = import_numpy_array(ENV_DATA_PATH)
    agent_population.clear_all_pops()
    agent_population.import_pops()
    paused = True  # Uncomment to pause the game after importing data

def update_game_logic():
    """
    Update the game logic, such as agent movements.
    """
    agent_population.take_action()
    agent_population.reset_turn()
    agent_population.Q_learning_update()

def update_display():
    """
    Update the game display, including rendering the world and agents.
    """
    world_rendering.world_render(agent_population)
    handle_mouse_hover()
    plot_graph()
    pygame.display.flip()

# Run the game loop
main_game_loop()