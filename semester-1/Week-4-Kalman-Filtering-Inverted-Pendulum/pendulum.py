import pygame, sys
import numpy as np
from pygame.locals import *

WINDOWDIMS = (1200, 600)
CARTDIMS = (50, 10)
PENDULUMDIMS = (6, 200)
GRAVITY = 0.13
REFRESHFREQ = 100
A_CART = 0.15

class InvertedPendulum(object):
    def __init__(self, windowdims, cartdims, penddims, gravity, a_cart):
        self.WINDOWWIDTH = windowdims[0]
        self.WINDOWHEIGHT = windowdims[1]

        self.CARTWIDTH = cartdims[0]
        self.CARTHEIGHT = cartdims[1]
        self.PENDULUMWIDTH = penddims[0]
        self.PENDULUMLENGTH = penddims[1]

        self.GRAVITY = gravity
        self.A_CART = a_cart
        self.Y_CART = 3 * self.WINDOWHEIGHT / 4
        self.reset_state()

    def reset_state(self):
        """initializes pendulum in upright state with small perturbation"""
        self.is_dead = False
        self.time = 0
        self.x_cart = self.WINDOWWIDTH / 2
        self.v_cart = 0
        # angle of pendulum (theta = 0 upright, omega positive into the screen)
        self.theta = np.random.uniform(-0.01,0.01)
        self.omega = 0

    def get_state(self):    
        return np.array([self.x_cart+np.random.normal(0,0.01),
                self.v_cart+np.random.normal(0,0.01), self.theta+np.random.normal(0,0.01), self.omega+np.random.normal(0,0.01),1])
    def get_real_state(self):
        return np.array([self.x_cart,
                self.v_cart, self.theta, self.omega,1])
    def set_state(self, state):
        is_dead, t, x, v, theta, omega = state
        self.is_dead = is_dead
        self.time = t
        self.x_cart = x
        self.v_cart = v
        self.theta = theta
        self.omega = omega

    def update_state_(self, action):
        """all the physics is here"""
        if self.is_dead:
            raise RuntimeError("tried to call update_state while state was dead")
        self.time += 1
        self.x_cart += self.v_cart
        # cart stops when it hits the wall
        if self.x_cart <= self.CARTWIDTH / 2 or self.x_cart >= self.WINDOWWIDTH - self.CARTWIDTH / 2:
            self.x_cart = self.WINDOWWIDTH / 2
            #self.v_cart = 0
        # term from angular velocity + term from motion of cart
        self.theta += self.omega + self.v_cart * np.cos(self.theta) / float(self.PENDULUMLENGTH)
        self.omega += self.GRAVITY * np.sin(self.theta) / float(self.PENDULUMLENGTH)
        self.v_cart+=action
        if abs(self.theta) >= np.pi / 2:
            self.is_dead = True
        return self.time, self.x_cart, self.v_cart, self.theta, self.omega
    def update_state(self, action):
        """all the physics is here"""
        if self.is_dead:
            raise RuntimeError("tried to call update_state while state was dead")
        self.time += 1
        self.x_cart += self.v_cart+np.random.normal(0,0.0001)
        # cart stops when it hits the wall
        if self.x_cart <= self.CARTWIDTH / 2 or self.x_cart >= self.WINDOWWIDTH - self.CARTWIDTH / 2:
            self.x_cart = self.WINDOWWIDTH / 2
            #self.v_cart = 0
        # term from angular velocity + term from motion of cart
        self.theta += self.omega + self.v_cart * np.cos(self.theta) / float(self.PENDULUMLENGTH)
        self.omega += self.GRAVITY * np.sin(self.theta) / float(self.PENDULUMLENGTH)
        self.v_cart+=action+np.random.normal(0,0.0001)
        if abs(self.theta) >= np.pi / 2:
            self.is_dead = True
        return self.time, self.x_cart, self.v_cart, self.theta, self.omega


class InvertedPendulumGame(object):
    def __init__(self, windowdims, cartdims, penddims,
                 gravity, a_cart, refreshfreq, pendulum = None):
        if pendulum is None:
            self.pendulum = InvertedPendulum(windowdims, cartdims, penddims, gravity, a_cart)
        else:
            self.pendulum = pendulum
        
        self.WINDOWWIDTH = windowdims[0]
        self.WINDOWHEIGHT = windowdims[1]

        self.CARTWIDTH = cartdims[0]
        self.CARTHEIGHT = cartdims[1]
        self.PENDULUMWIDTH = penddims[0]
        self.PENDULUMLENGTH = penddims[1]

        self.Y_CART = self.pendulum.Y_CART
        # self.time gives time in frames
        self.time = 0
        self.high_score = 0
        
        pygame.init()
        self.clock = pygame.time.Clock()
        # specify number of frames / state updates per second
        self.REFRESHFREQ = refreshfreq
        self.surface = pygame.display.set_mode(windowdims,0,32)
        pygame.display.set_caption('Inverted Pendulum Game')
        # array specifying corners of pendulum to be drawn
        self.static_pendulum_array = np.array(
            [[-self.PENDULUMWIDTH / 2, 0],
             [self.PENDULUMWIDTH / 2, 0],
             [self.PENDULUMWIDTH / 2, -self.PENDULUMLENGTH],
             [-self.PENDULUMWIDTH / 2, -self.PENDULUMLENGTH]]).T
        self.BLACK = (0,0,0)
        self.WHITE = (255,255,255)

    def draw_cart(self, x, theta):
        cart = pygame.Rect(x - self.CARTWIDTH // 2, self.Y_CART, self.CARTWIDTH, self.CARTHEIGHT)
        pygame.draw.rect(self.surface, self.BLACK, cart)
        pendulum_array = np.dot(self.rotation_matrix(theta), self.static_pendulum_array)
        pendulum_array += np.array([[x],[self.Y_CART]])
        pendulum = pygame.draw.polygon(self.surface, self.BLACK,
            ((pendulum_array[0,0],pendulum_array[1,0]),
             (pendulum_array[0,1],pendulum_array[1,1]),
             (pendulum_array[0,2],pendulum_array[1,2]),
             (pendulum_array[0,3],pendulum_array[1,3])))

    @staticmethod
    def rotation_matrix(theta):
        return np.array([[np.cos(theta), np.sin(theta)],
                         [-1 * np.sin(theta), np.cos(theta)]])

    def render_text(self, text, point, position = "center", fontsize = 48):
        font = pygame.font.SysFont(None, fontsize)
        text_render = font.render(text, True, self.BLACK, self.WHITE)
        text_rect = text_render.get_rect()
        if position == "center":
            text_rect.center = point
        elif position == "topleft":
            text_rect.topleft = point
        self.surface.blit(text_render, text_rect)

    def time_seconds(self):
        return self.time / float(self.REFRESHFREQ)

    def starting_page(self):
        self.surface.fill(self.WHITE)
        self.render_text("Inverted Pendulum",
                         (0.5 * self.WINDOWWIDTH, 0.4 * self.WINDOWHEIGHT))
        self.render_text("A Game by Adam Strandberg",
                         (0.5 * self.WINDOWWIDTH, 0.5 * self.WINDOWHEIGHT),
                         fontsize = 30)
        self.render_text("Press Enter to Begin",
                         (0.5 * self.WINDOWWIDTH, 0.7 * self.WINDOWHEIGHT),
                         fontsize = 30)
        pygame.display.update()

    def game_round(self,controller):
        np.random.seed(0)
        self.pendulum.reset_state()
        while not self.pendulum.is_dead:
            
            t, x, _, theta, _ = self.pendulum.update_state(controller(self.pendulum.get_state(),self.pendulum.get_real_state()))
            self.time = t    
            self.surface.fill(self.WHITE)
            self.draw_cart(x, theta)

            time_text = "t = {}".format(self.time_seconds())
            self.render_text(time_text, (0.1 * self.WINDOWWIDTH, 0.1 * self.WINDOWHEIGHT),
                             position = "topleft", fontsize = 40)
            
            pygame.display.update()
            self.clock.tick(self.REFRESHFREQ)
        if (self.time_seconds()) > self.high_score:
            self.high_score = self.time_seconds()

    def end_of_round(self):
        self.surface.fill(self.WHITE)
        self.draw_cart(self.pendulum.x_cart, self.pendulum.theta)
        self.render_text("Score: {}".format(self.time_seconds()),
                         (0.5 * self.WINDOWWIDTH, 0.3 * self.WINDOWHEIGHT))
        self.render_text("High Score : {}".format(self.high_score),
                         (0.5 * self.WINDOWWIDTH, 0.4 * self.WINDOWHEIGHT))
        self.render_text("(Enter to play again, ESC to exit)",
                         (0.5 * self.WINDOWWIDTH, 0.85 * self.WINDOWHEIGHT),
                         fontsize = 30)
        pygame.display.update()

    def game(self,controller):
        self.starting_page()
        
        while True:
            for event in pygame.event.get():
                if event.type == QUIT:
                    pygame.quit()
                    sys.exit()
                if event.type == KEYDOWN:
                    if event.key == K_RETURN:
                        self.game_round(controller)
                        self.end_of_round()
                    if event.key == K_ESCAPE:
                        pygame.quit()
                        sys.exit()

