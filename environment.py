
import gym
from gym import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import cv2
import sys
import random
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict
import torch.nn as nn 


class CustomDrawOptions(pymunk.pygame_util.DrawOptions):
    def __init__(self, surface):
        super().__init__(surface)

    def draw_shape(self, shape):
   
        if hasattr(shape, 'color') and shape.color is not None:
            color = shape.color
            self.set_color(color, shape)
        else:
            color = (200, 200, 200, 255) 
            self.set_color(color, shape)
        super().draw_shape(shape)


class RedSquareEnv(gym.Env):
    metadata = {'render.modes': ['background', 'rgb_array']}

    def __init__(self, data_generation=False, data_dir='dataset', gravity=5000, jump_impulse=500):
        super(RedSquareEnv, self).__init__()

        pygame.init()
        self.WIDTH, self.HEIGHT = 800, 600
        self.screen = pygame.Surface((self.WIDTH, self.HEIGHT))
        self.display_screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Red Square Environment")
        self.clock = pygame.time.Clock()

    
        self.space = pymunk.Space()
        self.space.gravity = (0, gravity)
        self.gravity = gravity  
        self.draw_options = CustomDrawOptions(self.screen)

 
        self.base_gravity = 900  
        self.base_jump_impulse = jump_impulse  
        self.scaled_jump_impulse = self.base_jump_impulse * math.sqrt(self.space.gravity[1] / self.base_gravity)

        self.create_slope()
        self.green_disks = []
        self.disk_timer = 0  

  
        self.obstacles = []

    
        self.max_disks = 10 
        self.disk_interval = random.randint(120, 300)  

        self.action_space = spaces.Discrete(4)  #
   
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(84, 84, 3), 
            dtype=np.uint8
        )

  
        self.can_jump = False
        self.mid_air_move_applied = False
        self.collision = False 
        self.surfaces = set()  

        self.state = None  
        self.previous_x = None  

        self.data_generation = data_generation
        self.data_dir = data_dir
        self.frame_count = 0

        if self.data_generation:
            if not os.path.exists(self.data_dir):
                os.makedirs(self.data_dir)
            self.annotations = []
        else:
           
            self.red_square_body, self.red_square_shape = self.create_red_square()

      
        self.create_random_disk()


        self.create_obstacles()

        self.setup_collision_handler()

    def create_slope(self):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)

        start_pos = (0, self.HEIGHT)
        end_pos = (self.WIDTH, self.HEIGHT - self.HEIGHT * 0.1)  
        shape = pymunk.Segment(body, start_pos, end_pos, 5)
        shape.friction = 0.9
        shape.elasticity = 0.8 
        shape.color = (255, 255, 255, 255)  
        shape.collision_type = 0  
        self.space.add(body, shape)

    def create_random_disk(self):
        mass = 1
        base_radius = 15
        radius = base_radius * random.uniform(0.8, 1.7)
        inertia = pymunk.moment_for_circle(mass, 0, radius)
        body = pymunk.Body(mass, inertia)

        position_x = random.uniform(600, self.WIDTH - 10)
        body.position = (position_x, 0)  
        shape = pymunk.Circle(body, radius)
        shape.friction = 0.9
        shape.elasticity = 0.8
    
        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            255
        )
        shape.color = color
        shape.collision_type = 3  
        self.space.add(body, shape)
        self.green_disks.append((body, shape))

    def create_red_square(self):
        mass = 1
        size = 30
        inertia = pymunk.moment_for_box(mass, (size, size))
        body = pymunk.Body(mass, inertia)

        body.position = (50, self.HEIGHT - 50)
        body.sleep_threshold = 0
        shape = pymunk.Poly.create_box(body, (size, size))
        shape.friction = 0.9
        shape.elasticity = 0.0
        shape.color = (255, 0, 0, 255)  
        shape.collision_type = 2  
        self.space.add(body, shape)
        return body, shape

    def setup_collision_handler(self):
  
        ground_handler = self.space.add_collision_handler(0, 2)  
        ground_handler.begin = self.begin_contact
        ground_handler.separate = self.end_contact

   
        obstacle_handler = self.space.add_collision_handler(1, 2)  
        obstacle_handler.begin = self.begin_contact
        obstacle_handler.separate = self.end_contact
        obstacle_handler.pre_solve = self.disk_collision  


        disk_handler = self.space.add_collision_handler(2, 3) 
        disk_handler.begin = self.begin_contact  
        disk_handler.separate = self.end_contact  

        disk_handler.pre_solve = self.disk_collision  

    def begin_contact(self, arbiter, space, data):
        shapes = arbiter.shapes
        if self.red_square_shape in shapes:
            idx = shapes.index(self.red_square_shape)
            other_shape = shapes[1 - idx]
            normal = arbiter.contact_point_set.normal

     
            if (idx == 0 and normal.y > 0.5) or (idx == 1 and normal.y < -0.5):
                self.surfaces.add(other_shape)
                self.can_jump = True
  
                self.mid_air_move_applied = False
        return True

    def end_contact(self, arbiter, space, data):
        shapes = arbiter.shapes
        if self.red_square_shape in shapes:
            idx = shapes.index(self.red_square_shape)
            other_shape = shapes[1 - idx]

            if other_shape in self.surfaces:
                self.surfaces.remove(other_shape)
                if not self.surfaces:
                    self.can_jump = False

        return True

    def disk_collision(self, arbiter, space, data):
        self.collision = True 
        return True

    def apply_action(self, action):
        if not self.data_generation:
            if action == 1:  
                if self.can_jump:
                    impulse = (-100, 0)
                    self.red_square_body.apply_impulse_at_world_point(impulse, self.red_square_body.position)
                elif not self.can_jump and not self.mid_air_move_applied:
                    impulse = (-100, 0)
                    self.red_square_body.apply_impulse_at_world_point(impulse, self.red_square_body.position)
                    self.mid_air_move_applied = False  
            elif action == 2: 
                if self.can_jump:
                    impulse = (100, 0)
                    self.red_square_body.apply_impulse_at_world_point(impulse, self.red_square_body.position)
                elif not self.can_jump and not self.mid_air_move_applied:
                    impulse = (100, 0)
                    self.red_square_body.apply_impulse_at_world_point(impulse, self.red_square_body.position)
                    self.mid_air_move_applied = True  
            elif action == 3 and self.can_jump:
                impulse = (0, -self.scaled_jump_impulse) 
                self.red_square_body.apply_impulse_at_world_point(impulse, self.red_square_body.position)
            else:
                pass  

    def get_observation(self):

        self.render(mode='rgb_array')
        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2))  


        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)

        # print('frame',frame.shape)
        frame = frame.astype(np.uint8)

        return frame  

    def create_obstacles(self):
     
        start_x = 0
        start_y = self.HEIGHT
        end_x = self.WIDTH
        end_y = self.HEIGHT - self.HEIGHT * 0.1


        slope_angle = math.atan2(end_y - start_y, end_x - start_x)

  
        num_obstacles = int(random.uniform(1, 5)) 
        for i in range(num_obstacles):
    
            t = random.uniform(0.3, 0.7)
            x = start_x + t * (end_x - start_x)
            y = start_y + t * (end_y - start_y)
            # print('check x, y', x, y)
            self.create_obstacle_on_slope(x, y, slope_angle)

    def create_obstacle_on_slope(self, x, y, slope_angle):
        body = pymunk.Body(body_type=pymunk.Body.STATIC)

 
        width = random.uniform(10, 40)
        height = random.uniform(20, 60) 


        shape = pymunk.Poly.create_box(body, (width, height))


        body.position = (x, y - height / 2)

        body.angle = slope_angle

        shape.friction = 0.0
        shape.elasticity = 0.0

        color = (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255),
            255
        )
        shape.color = color
        shape.collision_type = 1 

        self.space.add(body, shape)
        self.obstacles.append((body, shape))

    def reset(self):

        if not self.data_generation:
            self.space.remove(self.red_square_body, self.red_square_shape)
            self.red_square_body, self.red_square_shape = self.create_red_square()
            self.can_jump = False
            self.previous_x = self.red_square_body.position.x
        else:
            self.previous_x = None  

        self.mid_air_move_applied = False

        for body, shape in self.green_disks:
            self.space.remove(body, shape)
        self.green_disks = []
        self.disk_timer = 0  

        for body, shape in self.obstacles:
            self.space.remove(body, shape)
        self.obstacles = []
        self.create_obstacles() 

        self.create_random_disk()  

     
        self.disk_interval = random.randint(120, 300) 

        self.state = self.get_observation()
        self.collision = False  
        return self.state

    def step(self, action):

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                sys.exit()

        if self.data_generation:

            if hasattr(self, 'red_square_body') and hasattr(self, 'red_square_shape'):
                self.space.remove(self.red_square_body, self.red_square_shape)

            self.red_square_body, self.red_square_shape = self.create_red_square()

            iter = 0 
            while True:
                iter +=1
                x = random.uniform(50, self.WIDTH - 50)
                y = random.uniform(50, self.HEIGHT - 50)
                self.red_square_body.position = (x, y)

                overlapping = False
                for body, shape in self.obstacles:
                    if self.red_square_shape.shapes_collide(shape).points:
                        overlapping = True
                        break
                if iter == 100:
                    overlapping = False
                    self.red_square_body.position = (50, 50)
                
                if not overlapping:
                    break
                    
            for body, shape in self.obstacles:
                self.space.remove(body, shape)
            self.obstacles = []
            self.create_obstacles()  
          
            for body, shape in self.green_disks[:]:
                if body.position.y > self.HEIGHT + 100: 
                    self.space.remove(body, shape)
                    self.green_disks.remove((body, shape))

 
            self.disk_timer += 1
            if self.disk_timer >= self.disk_interval:
                if len(self.green_disks) < self.max_disks:
                    self.create_random_disk()
                self.disk_timer = 0
                self.disk_interval = random.randint(60,120) 
            self.space.step(2/60.0)


            self.state = self.get_observation()

      
            self.save_frame_and_annotation()

 
            done = False
            reward = 0
            info = {}
        else:

            self.apply_action(action)

     
            dt = 1/60.0 
            sub_steps = 5 
            for _ in range(sub_steps):
                self.space.step(dt / sub_steps)

   
            self.state = self.get_observation()

            for body, shape in self.green_disks[:]:
                if body.position.y > self.HEIGHT + 100:
                    self.space.remove(body, shape)
                    self.green_disks.remove((body, shape))


            self.disk_timer += 1
            if self.disk_timer >= self.disk_interval:
                if len(self.green_disks) < self.max_disks:
                    self.create_random_disk()
                self.disk_timer = 0
                self.disk_interval = random.randint(120, 300)

           
            done = False
            reward = -0.1  
            info = {}


            current_x = self.red_square_body.position.x
            if self.previous_x is not None:
                reward += (current_x - self.previous_x) * 0.125 * 5
            self.previous_x = current_x

   
            if self.collision:
                reward -= 1  
                self.collision = False 

            if self.red_square_body.position.x >= self.WIDTH - 2:
                done = True
                reward += 500  
                info['game_over'] = 'win'


            if self.red_square_body.position.x <= 0 or self.red_square_body.position.y > self.HEIGHT:
                done = True
                reward = -200  
                info['game_over'] = 'lose'

        return self.state, reward, done, info

    def save_frame_and_annotation(self):

        frame = pygame.surfarray.array3d(self.screen)
        frame = np.transpose(frame, (1, 0, 2)) 
        frame_path = os.path.join(self.data_dir, f"frame_{self.frame_count}.png")
        cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


        annotations = []


        red_pos = self.red_square_body.position
        red_size = 30 
        red_bbox = [red_pos.x - red_size/2, red_pos.y - red_size/2, red_size, red_size]
        annotations.append({'class': 'red_square', 'bbox': red_bbox})

        for body, shape in self.green_disks:
            pos = body.position
            radius = shape.radius
            bbox = [pos.x - radius, pos.y - radius, radius * 2, radius * 2]
            annotations.append({'class': 'disk', 'bbox': bbox})

        for body, shape in self.obstacles:
            verts = shape.get_vertices()
            verts = [v.rotated(body.angle) + body.position for v in verts]
            xs = [v.x for v in verts]
            ys = [v.y for v in verts]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
            annotations.append({'class': 'obstacle', 'bbox': bbox})

        self.annotations.append({'frame': frame_path, 'annotations': annotations})
        self.frame_count += 1

    def render(self, mode='background'):
        if mode == 'background':

            self.screen.fill((0, 0, 0))
            self.space.debug_draw(self.draw_options)
            self.display_screen.blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(60)
        elif mode == 'rgb_array':
    
            self.screen.fill((0, 0, 0))
            self.space.debug_draw(self.draw_options)
        else:
            raise NotImplementedError("Render mode '{}' not implemented.".format(mode))

    def close(self):
        if self.data_generation:
            import json
            annotation_file = os.path.join(self.data_dir, 'annotations.json')
            with open(annotation_file, 'w') as f:
                json.dump(self.annotations, f)
        pygame.quit()
