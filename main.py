import pygame
import numpy as np
import random
import time

from genetic_algorithm import GeneticAlgorithm
import neural_network


class Node:
    def __init__(self):
        self.current_position = None
        self.previous_node = None
        self.next_direction = None


class Snake:
    def __init__(self, row_size, column_size, rect_size):
        self.head = Node()
        self.tail = Node()
        self.score = 0
        self.size = 2
        self.row_size = row_size
        self.column_size = column_size
        self.rect_size = rect_size
        self.grid = np.zeros((self.row_size, self.column_size))
        self.keys = {
            "UP": (-1, 0),
            "DOWN": (1, 0),
            "LEFT": (0, -1),
            "RIGHT": (0, 1)
        }
        self.display_width = column_size * rect_size
        self.display_height = row_size * rect_size
        self.food_pos = tuple()

    def initialize(self):
        options = ["UP", "DOWN", "LEFT", "RIGHT"]

        tail_keys = {
            "UP": (1, 0),
            "DOWN": (-1, 0),
            "LEFT": (0, 1),
            "RIGHT": (0, -1)
        }

        head_keys = {
            "UP": (-2, 0),
            "DOWN": (2, 0),
            "LEFT": (0, -2),
            "RIGHT": (0, 2)
        }

        row = random.randint(0, self.row_size-1)
        column = random.randint(0, self.column_size-1)
        while (row, column) == (0, 0) or (row, column) == (0, self.column_size-1) or (row, column) == (self.row_size-1, 0) or (row, column) == (self.row_size-1, self.column_size-1):
            row = random.randint(0, self.row_size-1)
            column = random.randint(0, self.column_size-1)

        invalid = False
        dir = random.choice(options)
        update = head_keys[dir]
        (row_after_two_steps, column_after_two_steps) = (row + update[0], column + update[1])
        (tail_pos_row, tail_pos_column) = (row + tail_keys[dir][0], column + tail_keys[dir][1])
        temp = [row_after_two_steps, column_after_two_steps, tail_pos_row, tail_pos_column]

        if -1 in temp or -2 in temp or self.row_size in temp or self.row_size+1 in temp:
            invalid = True
        
        while invalid:
            options.remove(dir)
            dir = random.choice(options)
            update = head_keys[dir]
            (row_after_two_steps, column_after_two_steps) = (row + update[0], column + update[1])
            (tail_pos_row, tail_pos_column) = (row + tail_keys[dir][0], column + tail_keys[dir][1])
            temp = [row_after_two_steps, column_after_two_steps, tail_pos_row, tail_pos_column]
            if -1 in temp or -2 in temp or self.row_size in temp or self.row_size+1 in temp:
                invalid = True
            else:
                invalid = False

        self.head.current_position = (row, column)
        self.head.next_direction = dir
        self.tail.current_position = (tail_pos_row, tail_pos_column)
        self.tail.next_direction = dir
        self.tail.previous_node = self.head

        self.grid[self.head.current_position] = 1
        self.grid[self.tail.current_position] = 1

        self.grid[self.find_new_pos_for_food()] = 2
        return dir
    
    def collision_and_food_detection(self, direction):
        collision = False
        food = False

        if direction == "UP":
            if self.head.current_position[0] -1 < 0:
                collision = True
            elif self.grid[(self.head.current_position[0] -1, self.head.current_position[1])] == 1:
                collision = True
            elif self.grid[(self.head.current_position[0] -1, self.head.current_position[1])] == 2:
                food = True
        elif direction == "DOWN":
            if self.head.current_position[0] +1 == self.row_size:
                collision = True
            elif self.grid[(self.head.current_position[0] +1, self.head.current_position[1])] == 1:
                collision = True
            elif self.grid[(self.head.current_position[0] +1, self.head.current_position[1])] == 2:
                food = True
        elif direction == "LEFT":
            if self.head.current_position[1] -1 < 0:
                collision = True
            elif self.grid[(self.head.current_position[0], self.head.current_position[1] -1)] == 1:
                collision = True
            elif self.grid[(self.head.current_position[0], self.head.current_position[1] -1)] == 2:
                food = True
        elif direction == "RIGHT":
            if self.head.current_position[1] +1 == self.column_size:
                collision = True
            elif self.grid[(self.head.current_position[0], self.head.current_position[1] +1)] == 1:
                collision = True
            elif self.grid[(self.head.current_position[0], self.head.current_position[1] +1)] == 2:
                food = True

        return (collision, food)

    def update_grid(self, direction_taken, food):
        new_grid = np.zeros((self.row_size, self.column_size))

        pointer = self.head
        pointer.next_direction = direction_taken
        update = self.keys[direction_taken]
        new_position = (pointer.current_position[0] + update[0], pointer.current_position[1] + update[1])
        pointer.current_position = new_position
        new_grid[new_position] = 1

        current_tail_position = self.tail.current_position
        current_tail_next_direction = self.tail.next_direction

        pointer = self.tail
        
        for i in range(self.size - 1):
            update = self.keys[pointer.next_direction]
            new_position = (pointer.current_position[0] + update[0], pointer.current_position[1] + update[1])
            pointer.current_position = new_position
            pointer.next_direction = pointer.previous_node.next_direction
            new_grid[new_position] = 1
            pointer = pointer.previous_node

        if food:
            new_node = Node()
            new_node.current_position = current_tail_position
            new_node.next_direction = current_tail_next_direction
            new_node.previous_node = self.tail
            self.tail = new_node
            self.size += 1
            self.score += 1
            new_grid[self.find_new_pos_for_food()] = 2
        else:
            new_grid[self.food_pos] = 2

        self.grid = new_grid

    def find_new_pos_for_food(self):
        new_pos = tuple(int(random.random()*(self.row_size-1)) for _ in range(2))
        while(self.grid[new_pos] == 1):
            new_pos = tuple(int(random.random()*(self.row_size-1)) for _ in range(2))

        self.food_pos = new_pos
        return new_pos

    def draw_grid(self, win):
        for i in range(len(self.grid)):
            pygame.draw.line(win, (128, 128, 128), (0, i*self.rect_size), (self.display_width, i*self.rect_size))
            for j in range(len(self.grid[i])):
                pygame.draw.line(win, (128, 128, 128), (j*self.rect_size, 0), (j*self.rect_size, self.display_height))

    def draw_window(self, win):
        win.fill((0, 0, 0))
        pygame.draw.rect(win, (255, 0, 0), (0, 0, self.display_width, self.display_height), 4)

        for r in range(self.row_size):
            for c in range(self.column_size):
                if self.grid[r][c] == 1:
                    color = (255, 0, 0)
                elif self.grid[r][c] == 2:
                    color = (0, 255, 0)
                else:
                    color = (0, 0, 0)

                pygame.draw.rect(win, color, (c*self.rect_size, r*self.rect_size, self.rect_size, self.rect_size))

        self.draw_grid(win)
        pygame.display.update()


def wait_until_key_is_pressed():
    wait = True
    while wait:
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                wait = False


def main():
    row_size = 4
    column_size = 4
    rect_size = 50

    pygame.font.init()
    font = pygame.font.SysFont('comicsans', 60)

    pygame.display.set_caption('Snake')

    # snake = Snake(display_width, display_height, rect_size)
    # direction_taken = snake.initialize()
    # run = True
    
    # snake.draw_window(win)
    # time.sleep(0.5)

    population=100
    max_generation=100
    num_of_layers=3
    input_units_num=24
    hidden_units_num=20
    output_units_num=4
    mutation_rate=0.1

    ga = GeneticAlgorithm(
        population, max_generation, num_of_layers, input_units_num, hidden_units_num, output_units_num, mutation_rate
    )

    for i in range(max_generation):
        data_for_fitness = np.zeros((population, 2))
        max_steps_taken = 0
        print("generation %s", i)
        for j in range(population):
            snake = Snake(row_size, column_size, rect_size)
            win = pygame.display.set_mode((snake.display_width, snake.display_height))

            direction_taken = snake.initialize()

            snake.draw_window(win)
            time.sleep(0.5)
            game_over = False
            steps_taken = 0
            steps_taken_limit = 100
            score = 0

            while not game_over and steps_taken < steps_taken_limit:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False

                input_units = np.zeros(input_units_num)

                # FOOD
                # DISTANCE TO BODY
                # DISTANCE TO WALL


                # FOR RIGHT DIRECTION

                flag_for_food = 0
                flag_for_body = 0
                current_row, current_col = snake.head.current_position[0], snake.head.current_position[1] +1
                distance = 0
                while(current_col < snake.column_size):
                    distance += 1
                    if snake.grid[current_row][current_col] == 2 and flag_for_food == 0:
                        input_units[0] = distance
                        flag_for_food = 1
                    elif snake.grid[current_row][current_col] == 1 and flag_for_body == 0:
                        input_units[1] = distance
                        flag_for_body = 1
                    
                    current_col += 1

                input_units[2] = distance+1 # distance to wall

                if not flag_for_food:
                    input_units[0] = -1
                if not flag_for_body:
                    input_units[1] = -1
            

                # FOR UPPER RIGHT DIRECTION

                flag_for_food = 0
                flag_for_body = 0
                current_row, current_col = snake.head.current_position[0] -1, snake.head.current_position[1] +1
                distance = 0
                while(current_row >= 0 and current_col < snake.column_size):
                    distance += 1
                    if snake.grid[current_row][current_col] == 2 and flag_for_food == 0:
                        input_units[3] = distance
                        flag_for_food = 1
                    elif snake.grid[current_row][current_col] == 1 and flag_for_body == 0:
                        input_units[4] = distance
                        flag_for_body = 1
                    
                    current_row -= 1
                    current_col += 1

                input_units[5] = distance+1 # distance to wall

                if not flag_for_food:
                    input_units[3] = -1
                if not flag_for_body:
                    input_units[4] = -1

                
                # FOR UP DIRECTION
                flag_for_food = 0
                flag_for_body = 0
                current_row, current_col = snake.head.current_position[0] -1, snake.head.current_position[1]
                distance = 0
                while(current_row >= 0):
                    distance += 1
                    if snake.grid[current_row][current_col] == 2 and flag_for_food == 0:
                        input_units[6] = distance
                        flag_for_food = 1
                    elif snake.grid[current_row][current_col] == 1 and flag_for_body == 0:
                        input_units[7] = distance
                        flag_for_body = 1
                    
                    current_row -= 1

                input_units[8] = distance+1 # distance to wall

                if not flag_for_food:
                    input_units[6] = -1
                if not flag_for_body:
                    input_units[7] = -1


                # FOR UPPER LEFT DIRECTION
                flag_for_food = 0
                flag_for_body = 0
                current_row, current_col = snake.head.current_position[0] -1, snake.head.current_position[1] -1
                distance = 0
                while(current_row >= 0 and current_col >= 0):
                    distance += 1
                    if snake.grid[current_row][current_col] == 2 and flag_for_food == 0:
                        input_units[9] = distance
                        flag_for_food = 1
                    elif snake.grid[current_row][current_col] == 1 and flag_for_body == 0:
                        input_units[10] = distance
                        flag_for_body = 1
                    
                    current_row -= 1
                    current_col -= 1

                input_units[11] = distance+1 # distance to wall

                if not flag_for_food:
                    input_units[9] = -1
                if not flag_for_body:
                    input_units[10] = -1


                # FOR LEFT DIRECTION
                flag_for_food = 0
                flag_for_body = 0
                current_row, current_col = snake.head.current_position[0], snake.head.current_position[1] -1
                distance = 0
                while(current_col >= 0):
                    distance += 1
                    if snake.grid[current_row][current_col] == 2 and flag_for_food == 0:
                        input_units[12] = distance
                        flag_for_food = 1
                    elif snake.grid[current_row][current_col] == 1 and flag_for_body == 0:
                        input_units[13] = distance
                        flag_for_body = 1
                    
                    current_col -= 1

                input_units[14] = distance+1 # distance to wall

                if not flag_for_food:
                    input_units[12] = -1
                if not flag_for_body:
                    input_units[13] = -1


                # FOR LOWER LEFT DIRECTION
                flag_for_food = 0
                flag_for_body = 0
                current_row, current_col = snake.head.current_position[0] +1, snake.head.current_position[1] -1
                distance = 0
                while(current_row < snake.row_size and current_col >= 0):
                    distance += 1
                    if snake.grid[current_row][current_col] == 2 and flag_for_food == 0:
                        input_units[15] = distance
                        flag_for_food = 1
                    elif snake.grid[current_row][current_col] == 1 and flag_for_body == 0:
                        input_units[16] = distance
                        flag_for_body = 1
                    
                    current_row += 1
                    current_col -= 1

                input_units[17] = distance+1 # distance to wall

                if not flag_for_food:
                    input_units[15] = -1
                if not flag_for_body:
                    input_units[16] = -1


                # FOR DOWN DIRECTION
                flag_for_food = 0
                flag_for_body = 0
                current_row, current_col = snake.head.current_position[0] +1, snake.head.current_position[1]
                distance = 0
                while(current_row < snake.row_size):
                    distance += 1
                    if snake.grid[current_row][current_col] == 2 and flag_for_food == 0:
                        input_units[18] = distance
                        flag_for_food = 1
                    elif snake.grid[current_row][current_col] == 1 and flag_for_body == 0:
                        input_units[19] = distance
                        flag_for_body = 1
                    
                    current_row += 1

                input_units[20] = distance+1 # distance to wall

                if not flag_for_food:
                    input_units[18] = -1
                if not flag_for_body:
                    input_units[19] = -1


                # FOR LOWER RIGHT DIRECTION
                flag_for_food = 0
                flag_for_body = 0
                current_row, current_col = snake.head.current_position[0] +1, snake.head.current_position[1] +1
                distance = 0
                while(current_row < snake.row_size and current_col < snake.column_size):
                    distance += 1
                    if snake.grid[current_row][current_col] == 2 and flag_for_food == 0:
                        input_units[21] = distance
                        flag_for_food = 1
                    elif snake.grid[current_row][current_col] == 1 and flag_for_body == 0:
                        input_units[22] = distance
                        flag_for_body = 1
                    
                    current_row += 1
                    current_col += 1

                input_units[23] = distance+1 # distance to wall

                if not flag_for_food:
                    input_units[21] = -1
                if not flag_for_body:
                    input_units[22] = -1

                direction_to_be_taken = neural_network.compute(input_units, ga.weights1[j], ga.weights2[j])
                if direction_to_be_taken == "RIGHT":
                    if not direction_taken == "LEFT":
                        direction_taken = direction_to_be_taken
                elif direction_to_be_taken == "UP":
                    if not direction_taken == "DOWN":
                        direction_taken = direction_to_be_taken
                elif direction_to_be_taken == "LEFT":
                    if not direction_taken == "RIGHT":
                        direction_taken = direction_to_be_taken
                elif direction_to_be_taken == "DOWN":
                    if not direction_taken == "UP":
                        direction_taken = direction_to_be_taken

                print(direction_taken)

                (collision, food) = snake.collision_and_food_detection(direction_taken)
                if collision:
                    game_over = True
                    break
                if food:
                    score += 1
                    steps_taken_limit = steps_taken + 100

                steps_taken += 1
                snake.update_grid(direction_taken, food)
                snake.draw_window(win)
                time.sleep(0.2)

            if steps_taken > max_steps_taken:
                max_steps_taken = steps_taken

            data_for_fitness[j] = [score, steps_taken]

        ga.create_new_generation(data_for_fitness, max_steps_taken)


    # while run:
    #     for event in pygame.event.get():
    #         if event.type == pygame.QUIT:
    #             run = False

    #         if event.type ==  pygame.KEYDOWN:
    #             if event.key == pygame.K_LEFT:
    #                 if not direction_taken == "RIGHT":
    #                     direction_taken = "LEFT"
    #             if event.key == pygame.K_RIGHT:
    #                 if not direction_taken == "LEFT":
    #                     direction_taken = "RIGHT"
    #             if event.key == pygame.K_UP:
    #                 if not direction_taken == "DOWN":
    #                     direction_taken = "UP"
    #             if event.key == pygame.K_DOWN:
    #                 if not direction_taken == "UP":
    #                     direction_taken = "DOWN"

                
    #     (collision, food) = snake.collision_and_food_detection(direction_taken)
    #     if collision:
    #         label = font.render('Game Over', 1, (255, 255, 255))
    #         win.blit(label, (display_width/2 - (label.get_width()/2), 120))
    #         wait_until_key_is_pressed()
    #         run = False
    #         break

    #     snake.update_grid(direction_taken, food)
    #     snake.draw_window(win)
    #     time.sleep(0.1)


main()
