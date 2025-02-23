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
        self.rewardPerGeneration = []

        self.gridSize = gridSize
        self.snake = [(0, 0)]
        self.generateFoodInEmptySpace()
        self.move = moves.RIGHT_MOVE  # start moving out to the right

        self.direction = directions.RIGHT
        self.previousDirection = directions.RIGHT

        # self.cellSize = 20
        # self.windowSize = self.gridSize * self.cellSize
        # pygame.init()
        # self.screen = pygame.display.set_mode((self.windowSize, self.windowSize))
        # pygame.display.set_caption("Snake Game")
        # self.clock = pygame.time.Clock()

        self.action_space = spaces.Discrete(4)  # head can move up, down, left, right
        self.observation_space = spaces.Box(
            low=0, high=2, shape=(self.gridSize, self.gridSize), dtype=np.int8
        )

    def reset(self):
        # reset snake stack
        self.snake = [(0, 0)]
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
            reward, done = -10000000, True # discourage making invalid moves, huge negative score and immediately end game
        else:
            self.moveSnake()
            reward, done = self.calculateMoveAftermath()

        # if self.generation % 10000 < 1000:
        #     self.render()

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
        # here, turn make a 2d array using snake, food
        symbol_to_number = {" ": 0, "F": 1, "0": 2}
        return np.array(
            [
                [
                    symbol_to_number[self.getCoordinateSymbol((row, column))]
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
        self.previousMove = self.move
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
        outOfXBounds = self.getHeadXCoordinate() - 1 < 0
        outOfYBounds = self.getHeadYCoordinate() >= self.gridSize
        if outOfXBounds or outOfYBounds:
            return True
        return False

    def selfCollision(self):
        for index, coordinate in enumerate(self.snake):
            if index == 0:
                continue
            if coordinate == self.getHead():
                return True
        return False

    def calculateMoveAftermath(self):
        if len(self.snake) == self.gridSize**2:
            reward = 10000000
            done = True
        elif self.atFood():
            self.generateFoodInEmptySpace()
            reward = max(1000, len(self.snake)**2) # each food is worth more than the previous, encourage constant growth
            done = False
        elif self.outOfBounds() or self.selfCollision():
            self.snake.pop()
            reward = -10 * len(self.snake)
            done = True
        elif self.movedTowardsFood():
            self.snake.pop()
            reward = 0.5  # incentivise progress
            done = False
        else:
            self.snake.pop()
            reward = -0.2  # reduce stalling
            done = False

        return reward, done

    def generateFoodInEmptySpace(self):
        self.food = random.choice(self.getEmptySpaces())

    def getEmptySpaces(self):
        # return all coordinates where the snake isn't
        emptySpaces = []
        for xCoordinate in range(self.gridSize):
            for yCoordinate in range(self.gridSize):
                cell = (xCoordinate, yCoordinate)
                if cell not in self.snake:
                    emptySpaces.append(cell)

        return emptySpaces

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
        if coordinate == self.food:
            return "F"
        elif coordinate in self.snake:
            return "0"
        else:
            return " "

    def movedTowardsFood(self):
        newHead = self.getHead()
        oldHead = self.snake[1]
        if self.distanceFromFood(newHead) < self.distanceFromFood(oldHead):
            return True
        return False

    def distanceFromFood(self, cell):
        return abs(self.food[0] - cell[0]) + abs(self.food[1] - cell[1])

