import gym
import pygame
import random
import numpy as np
from .constants import directions, moves
from gym import spaces
from random import randint, choice
import math

shouldRender = True
renderCount = 1
renderInterval = 300

class SnakeEnv(gym.Env):
    def __init__(self, gridSize):
        super(SnakeEnv, self).__init__()

        self.generation = -1
        self.rewardPerGeneration = [0]

        self.gridSize = gridSize
        self.snake = [(self.gridSize // 2, self.gridSize // 2)]
        self.generateFoodInEmptySpace()
        self.move = moves.RIGHT_MOVE  # start moving out to the right

        self.direction = directions.RIGHT
        self.previousDirection = directions.RIGHT

        if shouldRender:
            self.cellSize = 50
            self.windowSize = self.gridSize * self.cellSize
            pygame.init()
            self.screen = pygame.display.set_mode((self.windowSize, self.windowSize))
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()

        self.action_space = spaces.Discrete(4)  # head can move up, down, left, right
        self.observation_space = spaces.Dict({
            "grid": spaces.Box(low=0, high=3, shape=(self.gridSize, self.gridSize), dtype=np.int8),
            "direction": spaces.Discrete(4),
            "previousDirection": spaces.Discrete(4),
            "distance_to_food": spaces.Box(low=0, high=self.gridSize * 2, shape=(4,), dtype=np.float32),
            "distance_to_obstacle": spaces.Box(low=0, high=self.gridSize, shape=(4,), dtype=np.float32),
            "snake_length": spaces.Box(low=2, high=self.gridSize**2, shape=(1,), dtype=np.int32),
            "relative_food_position": spaces.Box(low=-self.gridSize, high=self.gridSize, shape=(2,), dtype=np.float32),
            "body_proximity": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8),
            "tail_position": spaces.Box(low=-self.gridSize, high=self.gridSize, shape=(2,), dtype=np.float32),
            "direction_to_food": spaces.Box(low=0, high=1, shape=(4,), dtype=np.int8)
        })

    def reset(self):
        # reset snake stack
        self.snake = [(self.gridSize // 2, self.gridSize // 2)]
        self.generateFoodInEmptySpace()
        self.move = moves.RIGHT_MOVE  # start moving out to the right
        self.direction = directions.RIGHT
        self.previousDirection = directions.RIGHT
        self.generation += 1
        self.rewardPerGeneration.append(0)
        return self.getObservationSpace()

    def step(self, action):
        self.convertActionToMove(action)
        if not self.isValidMove():
            reward, done = (
                -10,
                True,
            )  # discourage making invalid moves, huge negative score and immediately end game
        else:
            self.moveSnake()
            reward, done = self.calculateMoveAftermath()

        if shouldRender and self.generation % renderInterval < renderCount:
            self.render()

        self.rewardPerGeneration[self.generation] += reward
        return self.getObservationSpace(), reward, done, {}

    def render(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

        pygame.display.set_caption("Snake Game - Generation " + str(self.generation))
        self.screen.fill((25, 25, 25))  # Dark background

        # Draw the grid
        for x in range(self.gridSize):
            for y in range(self.gridSize):
                rect = pygame.Rect(
                    y * self.cellSize, x * self.cellSize, self.cellSize, self.cellSize
                )
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 1)  # Grid lines

        # Draw the snake
        for i, segment in enumerate(self.snake):
            if segment == self.snake[0]:
                # Draw the head with a gradient and rounded corners
                head_rect = pygame.Rect(
                    segment[1] * self.cellSize,
                    segment[0] * self.cellSize,
                    self.cellSize,
                    self.cellSize,
                )
                pygame.draw.rect(self.screen, (0, 128, 255), head_rect, border_radius=10)
            else:
                # Draw the body with a gradient and rounded corners
                body_rect = pygame.Rect(
                    segment[1] * self.cellSize,
                    segment[0] * self.cellSize,
                    self.cellSize,
                    self.cellSize,
                )
                pygame.draw.rect(self.screen, (0, 255, 0), body_rect, border_radius=5)

        # Draw the food with a gradient and rounded corners
        food_rect = pygame.Rect(
            self.food[1] * self.cellSize,
            self.food[0] * self.cellSize,
            self.cellSize,
            self.cellSize,
        )
        pygame.draw.rect(self.screen, (255, 0, 0), food_rect, border_radius=15)

        # Add a shadow effect to the food
        shadow_rect = pygame.Rect(
            self.food[1] * self.cellSize + 5,
            self.food[0] * self.cellSize + 5,
            self.cellSize,
            self.cellSize,
        )
        pygame.draw.rect(self.screen, (128, 0, 0, 100), shadow_rect, border_radius=15)

        # Add a score display
        font = pygame.font.Font(None, 36)
        score_text = font.render(f"Score: {self.rewardPerGeneration[self.generation]}", True, (255, 255, 255))
        self.screen.blit(score_text, (10, 10))

        pygame.display.flip()
        self.clock.tick(6)  # Control game speed

    def getObservationSpace(self):
        symbolToNumber = {" ": 0, "F": 1, "H": 2, "B": 3}
        gridState = np.array(
            [
                [
                    symbolToNumber[self.getCoordinateSymbol((row, column))]
                    for column in range(self.gridSize)
                ]
                for row in range(self.gridSize)
            ],
            dtype=np.int8,
        )

        return {
            "grid": gridState,
            "direction": self.direction,
            "previousDirection": self.previousDirection,
            "distance_to_food": np.array(self.getDistanceToFood(), dtype=np.float32),
            "distance_to_obstacle": np.array(self.getDistanceToObstacle(), dtype=np.float32),
            "snake_length": np.array([len(self.snake)], dtype=np.int32),
            "relative_food_position": np.array(self.getRelativeFoodPosition(), dtype=np.float32),
            "body_proximity": np.array(self.getBodyProximity(), dtype=np.int8),
            "tail_position": np.array(self.getTailPosition(), dtype=np.float32),
            "direction_to_food": np.array(self.getDirectionToFood(), dtype=np.int8)
        }

    def getDistanceToFood(self):
        distances = []
        head = self.getHead()
        for direction in [moves.UP_MOVE, moves.DOWN_MOVE, moves.LEFT_MOVE, moves.RIGHT_MOVE]:
            new_pos = (head[0] + direction[0], head[1] + direction[1])
            distances.append(self.distanceFromFood(new_pos))
        return distances

    def getDistanceToObstacle(self):
        distances = []
        head = self.getHead()
        for direction in [moves.UP_MOVE, moves.DOWN_MOVE, moves.LEFT_MOVE, moves.RIGHT_MOVE]:
            distance = 0
            new_pos = head
            while True:
                new_pos = (new_pos[0] + direction[0], new_pos[1] + direction[1])
                if self.outOfBounds(new_pos) or new_pos in self.snake[1:]:
                    break
                distance += 1
            distances.append(distance)
        return distances

    def getRelativeFoodPosition(self):
        head = self.getHead()
        return (self.food[0] - head[0], self.food[1] - head[1])

    def getBodyProximity(self):
        head = self.getHead()
        proximity = []
        for direction in [moves.UP_MOVE, moves.DOWN_MOVE, moves.LEFT_MOVE, moves.RIGHT_MOVE]:
            new_pos = (head[0] + direction[0], head[1] + direction[1])
            proximity.append(int(new_pos in self.snake[1:]))
        return proximity

    def getTailPosition(self):
        tail = self.snake[-1]
        head = self.getHead()
        return (tail[0] - head[0], tail[1] - head[1])

    def getDirectionToFood(self):
        head = self.getHead()
        direction_to_food = []
        direction_to_food.append(int(self.food[0] < head[0]))  # Up
        direction_to_food.append(int(self.food[0] > head[0]))  # Down
        direction_to_food.append(int(self.food[1] < head[1]))  # Left
        direction_to_food.append(int(self.food[1] > head[1]))  # Right
        return direction_to_food

    def moveSnake(self):
        newHead = (
            self.getHeadXCoordinate() + self.move[0],
            self.getHeadYCoordinate() + self.move[1],
        )
        self.snake.insert(0, newHead)

    def getHead(self):
        return self.snake[0]

    def getHeadXCoordinate(self):
        return self.snake[0][0]

    def getHeadYCoordinate(self):
        return self.snake[0][1]

    def convertActionToMove(self, action):
        self.previousDirection = self.direction
        self.direction = action
        if action == directions.UP:
            self.move = moves.UP_MOVE
        elif action == directions.LEFT:
            self.move = moves.LEFT_MOVE
        elif action == directions.RIGHT:
            self.move = moves.RIGHT_MOVE
        elif action == directions.DOWN:
            self.move = moves.DOWN_MOVE

    def atFood(self):
        if self.snake[0] == self.food:
            return True
        return False

    def outOfBounds(self, cell=None):
        if cell is None:
            cell = self.getHead()
        outOfXBounds = cell[0] < 0 or cell[0] >= self.gridSize
        outOfYBounds = cell[1] < 0 or cell[1] >= self.gridSize
        return outOfXBounds or outOfYBounds

    def selfCollision(self):
        return len(set(self.snake)) != len(self.snake)

    def calculateMoveAftermath(self):
        if len(self.snake) == self.gridSize**2:
            reward = 10000000
            done = True
        elif self.atFood():
            self.generateFoodInEmptySpace()
            reward = 10 * len(self.snake)
            done = False
        elif self.outOfBounds() or self.selfCollision():
            self.snake.pop()
            reward = -10
            done = True
        elif self.movedTowardsFood():
            self.snake.pop()
            reward = .25  # incentivise progress
            done = False
        else:
            self.snake.pop()
            reward = -.5  # reduce stalling
            done = False

        return reward, done

    def generateFoodInEmptySpace(self):
        self.food = random.choice(self.getEmptySpaces())

    def getEmptySpaces(self):
        snake_set = set(self.snake)
        return [
            (x, y)
            for x in range(self.gridSize)
            for y in range(self.gridSize)
            if (x, y) not in snake_set
        ]

    def isValidMove(self):
        if (
            self.previousDirection == directions.LEFT
            and self.direction == directions.RIGHT
            or self.previousDirection == directions.RIGHT
            and self.direction == directions.LEFT
            or self.previousDirection == directions.UP
            and self.direction == directions.DOWN
            or self.previousDirection == directions.DOWN
            and self.direction == directions.UP
        ):
            return False
        return True

    def getCoordinateSymbol(self, coordinate):
        if coordinate == self.getHead():
            return "H" 
        elif coordinate in self.snake:
            return "B"
        elif coordinate == self.food:
            return "F" 
        else:
            return " "

    def movedTowardsFood(self):
        return self.distanceFromFood(self.getHead()) < self.distanceFromFood(
            self.snake[1]
        )

    def distanceFromFood(self, cell):
        return abs(self.food[0] - cell[0]) + abs(self.food[1] - cell[1])