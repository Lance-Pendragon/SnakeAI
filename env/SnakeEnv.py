import gym
import pygame
import random
import numpy as np
from .constants import directions, moves
from gym import spaces
from random import randint, choice
import math


class SnakeEnv(gym.Env):
    def __init__(self, gridSize):
        super(SnakeEnv, self).__init__()

        self.generation = -1
        self.rewardPerGeneration = [0]

        self.gridSize = gridSize
        self.snake = [(self.gridSize / 2, self.gridSize / 2)]
        self.generateFoodInEmptySpace()
        self.move = moves.RIGHT_MOVE  # start moving out to the right

        self.direction = directions.RIGHT
        self.previousDirection = directions.RIGHT

        self.cellSize = 20
        self.windowSize = self.gridSize * self.cellSize
        pygame.init()
        self.screen = pygame.display.set_mode((self.windowSize, self.windowSize))
        pygame.display.set_caption("Snake Game")
        self.clock = pygame.time.Clock()

        self.action_space = spaces.Discrete(4)  # head can move up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.gridSize, self.gridSize), dtype=np.int8
        )

    def reset(self):
        # reset snake stack
        self.snake = [(self.gridSize / 2, self.gridSize / 2)]
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
                -10000,
                True,
            )  # discourage making invalid moves, huge negative score and immediately end game
        else:
            self.moveSnake()
            reward, done = self.calculateMoveAftermath()

        self.render()

        self.rewardPerGeneration[self.generation] += reward
        return self.getObservationSpace(), reward, done, {}

    def render(self):
        pygame.display.set_caption("Snake Game - Generation " + str(self.generation))
        self.screen.fill((0, 0, 0))

        # Draw the snake
        for segment in self.snake:
            pygame.draw.rect(
                self.screen,
                (0, 255, 0),
                pygame.Rect(
                    segment[1] * self.cellSize,
                    segment[0] * self.cellSize,
                    self.cellSize,
                    self.cellSize,
                ),
            )

        # Draw the food
        pygame.draw.rect(
            self.screen,
            (255, 0, 0),
            pygame.Rect(
                self.food[1] * self.cellSize,
                self.food[0] * self.cellSize,
                self.cellSize,
                self.cellSize,
            ),
        )

        pygame.display.flip()
        self.clock.tick(10000)  # Control game speed

    def getObservationSpace(self):
        symbolToNumber = {" ": 0, "F": 1, "H": 2, "B": 3}
        return np.array(
            [
                [
                    symbolToNumber[self.getCoordinateSymbol((row, column))]
                    for column in range(self.gridSize)
                ]
                for row in range(self.gridSize)
            ],
            dtype=np.int8,
        )

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
        self.previousDirection = self.move
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

    def outOfBounds(self):
        outOfXBounds = self.getHeadXCoordinate() < 0 or self.getHeadXCoordinate() >= self.gridSize
        outOfYBounds = self.getHeadYCoordinate() < 0 or self.getHeadYCoordinate() >= self.gridSize

        if outOfXBounds or outOfYBounds:
            return True
        return False

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
            reward = -.5  # incentivise progress
            done = False
        else:
            self.snake.pop()
            reward = 0  # reduce stalling
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
