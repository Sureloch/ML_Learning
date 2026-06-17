# Example file showing a basic pygame "game loop"
import pygame
import math
import random

# pygame setup
pygame.init()
screen = pygame.display.set_mode((1280, 720))
center_point = [640, 360]

class Ball:
    location_x = 0
    location_y = 0
    velocity_x = 0
    velocity_y = 0
    is_moving_right = False
    
    def __init__(self,x0,y0,vx0,vy0,dir):
        self.location_x = x0
        self.location_y = y0
        self.velocity_x = vx0
        self.velocity_y = vy0
        self.is_moving_right = dir
        pass
    
    
circle_location_x = []
circle_location_y = []
#circle_location = [640,0]
circle_velocity_y = []
circle_velocity_x = []
is_circle_facing_right = []

friction_coefficient = 0.9
threshold = 20
clock = pygame.time.Clock()
dt = 0
radius = 10
rect_offset_y = 30
running = True
g = -1000

def calculate_distance(x1, y1, x2, y2, y_axis = False):
    if y_axis == True:
        return math.sqrt((y2- y1)**2)
    else:
        return math.sqrt((x2 - x1)**2 + (y2- y1)**2)

while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            circle_location_x.append(pygame.mouse.get_pos()[0])
            circle_location_y.append(pygame.mouse.get_pos()[1])
            circle_velocity_x.append(random.randrange(-600, 600 , 1))
            circle_velocity_y.append(0)
            if circle_velocity_x[-1] > 0:
                is_circle_facing_right.append(1)
            else:
                is_circle_facing_right.append(0)
            
            

    # fill the screen with a color to wipe away anything from last frame
    screen.fill("white")
    RED = (255, 0 , 0)
    BLUE = (0, 0, 255)
    # RENDER YOUR GAME HERE
    rect = pygame.Rect(0,center_point[1] + 50, 1280, 50)
    pygame.draw.rect(screen, RED, rect)

    if len(circle_location_y) > 0:
        for i in range(len(circle_location_x)):
            if circle_velocity_x[i] > 0.0:
                is_circle_facing_right[i] = 1
            else:
                is_circle_facing_right[i] = 0
            circle_velocity_y[i] += g * dt
            normal_force = g/100
            friction_force = abs(normal_force * friction_coefficient)
            
            circle_velocity_x[i] -= friction_force * dt
            circle_location_y[i] -= circle_velocity_y[i] * dt
            circle_location_x[i] += circle_velocity_x[i] * dt
            
            

            if circle_location_x[i] + radius > screen.width:
                    circle_velocity_x[i] *= - 1
                
            if circle_location_x[i] - radius < 0:
                    circle_velocity_x[i] *= - 1
                    
            if circle_location_y[i] - radius > center_point[1] + rect_offset_y:
                circle_location_y[i] = center_point[1] + rect_offset_y + radius
                if is_circle_facing_right[i] == 1:
                    circle_velocity_x[i] -= friction_force
                else:
                    circle_velocity_x[i] += friction_force
                # bounce only if moving downward
                if circle_velocity_y[i] < 0:
                    circle_velocity_y[i] *= -0.8

                    # stop tiny bouncing
                    if abs(circle_velocity_y[i]) < 15:
                        circle_velocity_y[i] = 0
                    if abs(circle_velocity_x[i]) < 15:
                        circle_velocity_x[i] = 0
                        
            

                
            if i % 2 == 0:
                circle = pygame.draw.circle(screen, BLUE, (circle_location_x[i], circle_location_y[i]), 10)
            else:
                circle = pygame.draw.circle(screen, RED, (circle_location_x[i], circle_location_y[i]), 10)
                
            other_circles = circle_location_x[i:]
            offset = len(circle_location_x) - len(other_circles)   # offset = i
            if len(other_circles) > 0:
                for j in range(len(other_circles)):
                    idx = j + offset   # index of the other circle
                    dist = calculate_distance(circle_location_x[i], circle_location_y[i],
                                            circle_location_x[idx], circle_location_y[idx])
                    if dist < 20 and dist != 0:
                        overlap = 20 - dist
                        # Swap velocities (elastic collision for equal masses)
                        tmp_x = circle_velocity_x[idx]
                        circle_velocity_x[idx] = circle_velocity_x[i]
                        circle_velocity_x[i] = tmp_x

                        tmp_y = circle_velocity_y[idx]
                        circle_velocity_y[idx] = circle_velocity_y[i]
                        circle_velocity_y[i] = tmp_y

                        # ----- Position correction (clipping) -----
                        # Direction from circle i to the other circle
                        dx = circle_location_x[idx] - circle_location_x[i]
                        dy = circle_location_y[idx] - circle_location_y[i]
                        # Unit vector
                        length = (dx**2 + dy**2)**0.5
                        if length > 0:
                            dx /= length
                            dy /= length
                            # Move each circle by half the overlap along the direction
                            correction = overlap / 2
                            circle_location_x[i] -= dx * correction
                            circle_location_y[i] -= dy * correction
                            circle_location_x[idx] += dx * correction
                            circle_location_y[idx] += dy * correction

                        





   
    
    pygame.display.flip()

    clock.tick(60)  # limits FPS to 60
    dt = clock.tick(60) / 1000

pygame.quit()