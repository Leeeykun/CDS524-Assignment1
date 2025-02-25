import random
import numpy as np
import pygame
import os

# Global constants for screen dimensions
WIDTH = 800  
HEIGHT = 600  

# Q-Learning parameters
LEARNING_RATE = 0.1  # Learning rate for Q-table updates, controls how quickly the Q-values adapt
DISCOUNT_FACTOR = 0.9  # Discount factor, balances immediate vs future rewards (0 to 1)
EPSILON = 0.3  # Initial exploration rate, probability of choosing a random action
EPSILON_DECAY = 0.98  # Decay rate for epsilon, reduces exploration over time
MIN_EPSILON = 0.05  # Minimum exploration rate, ensures some randomness persists

# Define state and action space
STATE_SIZE = 1000  # Total number of states: 10 (y position) x 10 (gap relative) x 10 (velocity)
ACTION_SIZE = 2  # Number of possible actions: 0 (do nothing), 1 (jump)

# File path for saving/loading the Q-table
Q_TABLE_FILE = "q_table.npy"

# Initialize or load the Q-table
# Check if a saved Q-table exists; if so, load it, otherwise initialize a new one with zeros
if os.path.exists(Q_TABLE_FILE):
    q_table = np.load(Q_TABLE_FILE)
    print(f"Loaded Q-table from {Q_TABLE_FILE}")  # Confirm loading of previous Q-table
else:
    q_table = np.zeros((STATE_SIZE, ACTION_SIZE))  # Create a new Q-table with all zeros
    print("Initialized new Q-table")

# Bird class representing the player-controlled object
class Bird:
    def __init__(self):
        self.reset()  # Initialize bird with default values

    def reset(self):
        # Reset bird to starting position and state
        self.x = WIDTH // 4  # Horizontal position, 1/4 of screen width
        self.y = HEIGHT // 2  # Vertical position, middle of screen
        self.velocity = 0  # Initial vertical velocity
        self.size = 10  # Radius of the bird in pixels
        self.gravity = 1.0  # Gravity acceleration affecting downward movement
        self.jump_velocity = -6  # Velocity applied when jumping (negative = upward)

    def update(self):
        # Update bird's position based on velocity and gravity
        self.velocity += self.gravity  # Increase velocity due to gravity
        self.y += self.velocity  # Update y position with current velocity
        # Ensure bird stays within screen bounds
        if self.y < 0:
            self.y = 0
            self.velocity = 0  # Stop upward movement at top
        if self.y > HEIGHT:
            self.y = HEIGHT
            self.velocity = 0  # Stop downward movement at bottom

    def jump(self):
        # Perform a jump by setting velocity to a negative value
        self.velocity = self.jump_velocity

    def draw(self, screen):
        # Draw the bird as a blue circle on the screen
        pygame.draw.circle(screen, (0, 0, 255), (int(self.x), int(self.y)), self.size)

# Pipe class representing obstacles in the game
class Pipe:
    def __init__(self):
        # Initialize pipe with random gap position
        self.x = WIDTH  # Start at right edge of screen
        self.gap = 150  # Vertical gap size between top and bottom pipes
        self.gap_y = random.randint(100, HEIGHT - 100 - self.gap)  # Random y position of gap top
        self.width = 50  # Width of the pipe
        self.passed = False  # Flag to track if bird has passed this pipe
        self.speed = 4  # Horizontal speed of pipe movement (pixels per frame)

    def update(self):
        # Move pipe leftward across the screen
        self.x -= self.speed

    def draw(self, screen):
        # Draw top and bottom pipes as green rectangles
        pygame.draw.rect(screen, (0, 255, 0), (self.x, 0, self.width, self.gap_y))  # Top pipe
        pygame.draw.rect(screen, (0, 255, 0), (self.x, self.gap_y + self.gap, self.width, HEIGHT - self.gap_y - self.gap))  # Bottom pipe

    def collide(self, bird):
        # Check for collision between bird and pipe
        bird_rect = pygame.Rect(bird.x - bird.size, bird.y - bird.size, bird.size * 2, bird.size * 2)  # Bird's bounding box
        top_pipe = pygame.Rect(self.x, 0, self.width, self.gap_y)  # Top pipe rectangle
        bottom_pipe = pygame.Rect(self.x, self.gap_y + self.gap, self.width, HEIGHT - self.gap_y - self.gap)  # Bottom pipe rectangle
        collide_top = bird_rect.colliderect(top_pipe)  # Check collision with top pipe
        collide_bottom = bird_rect.colliderect(bottom_pipe)  # Check collision with bottom pipe
        if collide_top or collide_bottom:
            # Print collision details for debugging
            print(f"Collision - Top: {collide_top}, Bottom: {collide_bottom}, Bird: ({bird.x}, {bird.y}), Pipe: ({self.x}, {self.gap_y})")
        return collide_top or collide_bottom

# Function to get discrete state representation
def get_state(bird, pipes):
    # Find the next pipe to the right of the bird
    next_pipe = next((p for p in pipes if p.x + p.width > bird.x), None)
    if not next_pipe:
        return 0  # Default state if no pipe ahead
    # Discretize bird's y position into 10 levels (0 to 9)
    bird_y = min(max(int(bird.y / 60), 0), 9)
    # Calculate center of the next pipe's gap
    gap_center = next_pipe.gap_y + next_pipe.gap // 2
    # Discretize relative distance to gap center into 10 levels (0 to 9)
    gap_relative = min(max(int((bird.y - gap_center) / 30), -5), 4) + 5
    # Discretize velocity into 10 levels (0 to 9), mapping [-10, 10] to [0, 9]
    velocity = min(max(int((bird.velocity + 10) / 2), 0), 9)
    # Combine into a single state index
    return bird_y * 100 + gap_relative * 10 + velocity

# Function to reset the game state
def reset_game(bird, pipes):
    bird.reset()  # Reset bird to initial position
    pipes.clear()  # Remove all pipes
    pipes.append(Pipe())  # Add a new pipe
    return 0  # Reset score to 0

# Function to save the Q-table to a file
def save_q_table():
    np.save(Q_TABLE_FILE, q_table)  # Save Q-table as a .npy file
    print(f"Saved Q-table to {Q_TABLE_FILE}")  # Confirm save operation

# Main game loop
def game_loop():
    global EPSILON
    
    # Initialize Pygame and set up the display
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flappy Bird with Q-Learning")
    clock = pygame.time.Clock()
    
    # Define colors
    WHITE = (255, 255, 255)  # Background color
    BLACK = (0, 0, 0)  # Text color
    
    # Initialize game objects
    bird = Bird()
    pipes = [Pipe()]
    score = 0  
    high_score = 0  
    font = pygame.font.SysFont("arial", 24)  
    episode = 0  
    total_reward = 0  
    pass_distances = []  
    current_action = "Down"  # Default action display (updated based on agent's choice)
    
    running = True
    while running:
        # Handle Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                save_q_table()  # Save Q-table when quitting
        
        screen.fill(WHITE)  # Clear screen with white background
        
        # Q-Learning decision-making
        state = get_state(bird, pipes)  # Get current state
        if random.random() < EPSILON:
            action = random.randint(0, ACTION_SIZE - 1)  # Random action (explore)
        else:
            action = np.argmax(q_table[state])  # Best action based on Q-table (exploit)
        
        # Update action display based on agent's choice
        if action == 1:
            bird.jump()  # Jump if action is 1
            current_action = "Up"  # Agent chose to jump (upward movement)
        else:
            current_action = "Down"  # Agent chose to do nothing (downward due to gravity)
        bird.update()  # Update bird's position
        
        # Get the next pipe ahead of the bird
        next_pipe = next((p for p in pipes if p.x + p.width > bird.x), pipes[0])
        gap_center = next_pipe.gap_y + next_pipe.gap // 2  # Center of the gap
        
        # Update pipes and handle scoring
        for pipe in pipes[:]:
            pipe.update()  # Move pipe left
            if pipe.x + pipe.width < 0:
                pipes.remove(pipe)  # Remove pipe if off-screen
            if pipe.x + pipe.width < bird.x and not pipe.passed:
                pipe.passed = True  # Mark pipe as passed
                score += 1  # Increment score
                high_score = max(high_score, score)  # Update high score
                pass_distances.append(abs(bird.y - gap_center))  # Record distance from gap center
        
        # Spawn new pipe if needed
        if pipes[-1].x < WIDTH - 300:
            pipes.append(Pipe())
        
        # Calculate reward based on bird's position and actions
        reward = 0
        distance_to_gap_center = abs(bird.y - gap_center)
        if distance_to_gap_center < 30:
            reward += 50 - distance_to_gap_center  # Reward for being close to gap center
        elif distance_to_gap_center > 75:
            reward -= (distance_to_gap_center - 75) ** 2 / 100  # Penalty for being far from gap
        if any(pipe.collide(bird) for pipe in pipes):
            reward = -50  # Penalty for collision
            total_reward += reward
            score = reset_game(bird, pipes)  # Reset game on collision
            episode += 1  # Increment episode count
        elif any(pipe.passed and pipe.x + pipe.width < bird.x for pipe in pipes):
            reward += 50  # Reward for passing a pipe
        
        # Update Q-table with the new experience
        next_state = get_state(bird, pipes)
        q_table[state, action] = (1 - LEARNING_RATE) * q_table[state, action] + \
                                 LEARNING_RATE * (reward + DISCOUNT_FACTOR * np.max(q_table[next_state]))
        
        total_reward += reward  # Accumulate total reward
        EPSILON = max(MIN_EPSILON, EPSILON * EPSILON_DECAY)  # Decay exploration rate
        
        # Render game elements
        bird.draw(screen)
        for pipe in pipes:
            pipe.draw(screen)
        # UI text elements
        score_text = font.render(f"Score: {score}", True, BLACK)
        high_score_text = font.render(f"High Score: {high_score}", True, BLACK)
        episode_text = font.render(f"Episodes: {episode}", True, BLACK)
        action_text = font.render(f"Action: {current_action}", True, BLACK)  # Display agent's action
        reward_text = font.render(f"Total Reward: {int(total_reward)}", True, BLACK)  # Display total reward
        # Position text on screen
        screen.blit(score_text, (10, 10))  # Display current score
        screen.blit(high_score_text, (10, 40))  # Display high score
        screen.blit(episode_text, (10, 70))  # Display episode count
        screen.blit(action_text, (10, 100))  # Display action below episode count
        screen.blit(reward_text, (10, 130))  # Display total reward below action
        
        pygame.display.flip()  # Update the display
        clock.tick(60)  # Cap frame rate at 60 FPS
        
        # Print logs and save Q-table every 30 episodes
        if episode % 30 == 0 and episode > 0 and score == 0:
            avg_pass_distance = sum(pass_distances) / len(pass_distances) if pass_distances else 0
            print(f"Episode {episode}, Score: {score}, Total Reward: {total_reward}, Epsilon: {EPSILON:.4f}, High Score: {high_score}, Q Mean: {np.mean(q_table):.2f}")
            print(f"Avg Pass Distance: {avg_pass_distance:.2f}")
            initial_state = get_state(bird, pipes)
            print(f"Initial State Q-values: {q_table[initial_state]}")
            save_q_table()  # Save Q-table to file

    pygame.quit()  # Clean up and exit Pygame

if __name__ == "__main__":
    game_loop()