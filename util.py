import math
import random


def rand(low, high):
	""" Return a continuous pseudo-random value between two values """
	return low + (high - low) * random.random()


def dist(p1, p2):
	""" Return euclidean distance between two points """
	return math.sqrt(dist_sqr(p1, p2))


def dist_sqr(p1, p2):
	""" Return euclidean distance^2 between two points; saves on computation """
	dx = p1[0] - p2[0]
	dy = p1[1] - p2[1]
	return dx*dx + dy*dy


def sigmoid(x):
	""" Return standard sigmoid function (~math.tanh) """
	return 1.0 / (1.0 + math.exp(-x))


def clamp(value, min_, max_):
	""" Limit and return the passed value between the passed limits """
	if value < min_:
		value = min_
	elif value > max_:
		value = max_
	return value


def int_tuple(tuple_):
	""" Return a tuple with equivalent integer components """
	return (int(tuple_[0]), int(tuple_[1]))


def sgn(value):
	""" Return -1, 0, or 1 if value < 0, == 0, > 0 """
	return 1 - (1 * value == 0) - (2 * value < 0)
