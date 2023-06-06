import pygame
import numpy as np
from network import NeuralNetwork
import cv2
import math

FRAME_RATE = 360
SCREEN_WIDTH = 1080
SCREEN_HEIGHT = 720
BACKGROUND_COLOUR = (40,40,40)
PASTEL_GREEN = (193,225,193)
WHITE = (255,255,255)

pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Digit Recognition")
clock = pygame.time.Clock()

class Canvas:
    def __init__(self, pixels = None):
        self.pixels_each_side = 28
        self.pixel_side_length = 20
        self.pixels = self.initialise_pixels(pixels)
        self.x_offset = 80
        self.y_offset = 80

    def initialise_pixels(self, pixels=None):
        if pixels is None:
            return np.zeros((self.pixels_each_side,self.pixels_each_side)) # pixels can take a value from 0 to 255
        else:
            return pixels

    def set_pixel(self, coordinates, newvalue: int):
        self.pixels[coordinates[1]][coordinates[0]] = newvalue

    def get_coordinates(self, x, y):
        if (x > self.x_offset + self.pixel_side_length * self.pixels_each_side or x < self.x_offset) or (y > self.y_offset + self.pixel_side_length * self.pixels_each_side or y < self.y_offset):
            return None
        else:
            return ((x - self.x_offset) // self.pixel_side_length, (y - self.y_offset) // self.pixel_side_length)

    def draw_on_canvas(self, mouse_x_pos, mouse_y_pos, mode):
        coords = self.get_coordinates(mouse_x_pos, mouse_y_pos)
        if coords is not None:
            if mode == "DRAW":
                self.pixels[coords[1]][coords[0]] = 255
            elif mode == "ERASE":
                self.pixels[coords[1]][coords[0]] = 0

    def clear(self):
        self.pixels = self.initialise_pixels()

    def calculate_centre_of_mass(self, matrix):
        """ Takes a 2D matrix as input and returns the coordinates of centre of mass"""
        totalmass = 0
        sum_mx = 0
        sum_my = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):
                totalmass += matrix[i][j]
                sum_mx += j * matrix[i][j]
                sum_my += i * matrix[i][j]
        
        if totalmass == 0:
            return None
        else:
            return (int(round(sum_mx / totalmass, 0)), int(round(sum_my / totalmass,0)))

    def get_shift(self, resized_matrix):
        shifts = self.calculate_centre_of_mass(resized_matrix)
        if shifts is not None:
            cx, cy = shifts
            return (15-cx, 15-cy)
        else:
            return None
    
    # This is taken from https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
    def remove_zero(self):
        # removes all the zero rows and columns
        resized_pixels = self.pixels.copy()
        total = sum([sum(resized_pixel_row) for resized_pixel_row in resized_pixels])

        if total != 0:
            while np.sum(resized_pixels[0]) == 0:
                resized_pixels = resized_pixels[1:]
            while np.sum(resized_pixels[-1]) == 0:
                resized_pixels = resized_pixels[:-1]
            while np.sum(resized_pixels[:,0]) == 0:
                resized_pixels = np.delete(resized_pixels,0,1)
            while np.sum(resized_pixels[:,-1]) == 0:
                resized_pixels = np.delete(resized_pixels,-1,1)

        return resized_pixels
    
    # This is taken from https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
    def resize(self, matrix):
        rows, cols = matrix.shape
        if rows > cols:
            factor = 20.0/rows
            rows = 20
            cols = int(round(cols*factor))
            matrix = cv2.resize(matrix, (cols,rows))
        else:
            factor = 20.0/cols
            cols = 20
            rows = int(round(rows*factor))
            matrix = cv2.resize(matrix, (cols, rows))

        colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
        rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
        matrix = np.lib.pad(matrix,(rowsPadding,colsPadding),'constant')

        return matrix

    # This is taken from https://medium.com/@o.kroeger/tensorflow-mnist-and-your-own-handwritten-digits-4d1cd32bbab4
    def shift_by_centre_of_mass(self, matrix):
        shifts = self.get_shift(matrix)
        if shifts is not None:
            shift_x, shift_y = shifts
            rows,cols = matrix.shape
            M = np.float32([[1,0,shift_x],[0,1,shift_y]])
            matrix = cv2.warpAffine(matrix,M,(cols,rows))
        return matrix

    def get_new_pixels(self):
        nozero = self.remove_zero()
        resized = self.resize(nozero)
        shifted = self.shift_by_centre_of_mass(resized)
        return shifted

    def display(self):
        for i in range(self.pixels_each_side):
            for j in range(self.pixels_each_side):
                brightness = self.pixels[j][i]
                pygame.draw.rect(screen, (brightness, brightness, brightness), (self.x_offset + i * self.pixel_side_length, self.y_offset + j * self.pixel_side_length, self.pixel_side_length, self.pixel_side_length))
                
    def display_test(self):

        shifted = self.get_new_pixels()

        for i in range(28):
            for j in range(28):
                x_offset = 700
                y_offset = 50
                length = 12
                brightness = shifted[j][i]
                pygame.draw.rect(screen, (brightness, brightness, brightness), (x_offset + i * length, y_offset + j * length, length, length))

class Game:
    def __init__(self, canvas:Canvas, neural_network:NeuralNetwork):
        self.canvas = canvas
        self.mode = "IDLE" # Can take "IDLE", "DRAW" or "ERASE"
        self.neural_network = neural_network
        self.results = None

    def identify(self):
        input_vector = self.canvas.get_new_pixels().flatten()
        output_vector = self.neural_network.forward_propogate(input_vector)
        results = [(i, round(output_vector[i], 3)) for i in range(10)]
        results.sort(key = lambda x: x[1], reverse = True)
        return results

    def display_results(self):
        font = pygame.font.Font(None, 55)
        for i in range(10):
            if i == 0:
                text = font.render(str(self.results[i][0]) + ": " + str(self.results[i][1]), True, PASTEL_GREEN)
            else:
                text = font.render(str(self.results[i][0]) + ": " + str(self.results[i][1]), True, WHITE)
            screen.blit(text, (780, 120 + i * 50))

    def simulate(self):
        while True:
            clock.tick(FRAME_RATE)
            screen.fill(BACKGROUND_COLOUR)

            x_pos, y_pos = pygame.mouse.get_pos()
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    pygame.quit()
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        pygame.quit()
                        break
                    if event.key == pygame.K_SPACE:
                        self.canvas.clear()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.mode = "DRAW"
                    if event.button == 3:
                        self.mode = "ERASE"
                if event.type == pygame.MOUSEBUTTONUP:
                    self.mode = "IDLE"

            self.canvas.draw_on_canvas(x_pos, y_pos, self.mode)
            self.canvas.display()
            
            #self.canvas.display_test()

            self.results = self.identify()
            self.display_results()

            font2 = pygame.font.Font(None, 30)
            instruction_text = font2.render("Left Click: Draw | Right Click: Erase | Space: Clear", True, WHITE)
            screen.blit(instruction_text, (112, 660))

            pygame.display.update()
