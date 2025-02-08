import cv2
import random
import numpy as np
from .config import config

particles = []

def create_explosion(x, y):
    global particles
    count = config["particles"]["count"]
    for _ in range(count):
        angle = random.uniform(0, 2 * np.pi)
        speed = random.uniform(config["particles"]["speed_min"], config["particles"]["speed_max"])
        color = [random.randint(100, 255) for _ in range(3)]
        size = random.randint(config["particles"]["size_min"], config["particles"]["size_max"])
        life = random.randint(config["particles"]["life_min"], config["particles"]["life_max"])
        gravity = random.uniform(config["particles"]["gravity_min"], config["particles"]["gravity_max"])
        particles.append([x, y, np.cos(angle) * speed, np.sin(angle) * speed, color, life, size, gravity])

def update_particles(frame):
    global particles
    new_particles = []
    for p in particles:
        x, y, vx, vy, color, life, size, gravity = p
        x += vx
        y += vy
        vy += gravity
        life -= 1
        if life > 0:
            cv2.circle(frame, (int(x), int(y)), size, color, -1)
            new_particles.append([x, y, vx, vy, color, life, size, gravity])
    particles = new_particles
    return frame
