from collections import namedtuple
import numpy as np

DataPoint = namedtuple('DataPoint', ['xs', 'y'])


def func1(x1, x2):
    return x1 + 2 * x2


def make_datapoint(xs, yfunc):
    return DataPoint(xs, yfunc(*xs))


p1 = make_datapoint([-2,0], func1)
p2 = make_datapoint([-1,0], func1)
p3 = make_datapoint([1,0], func1)


print(p1)
