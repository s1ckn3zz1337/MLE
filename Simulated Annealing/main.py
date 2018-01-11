import random
import sys
import math
import datetime
from random import shuffle


def init_distance_array(count, dist_range):
    distance_array = [[0 for x in xrange(0, count)] for y in xrange(0, count)]
    for x, valueX in enumerate(distance_array):
        for y, valueY in enumerate(valueX):
            if x == y:
                distance_array[x][y] = 0
            else:
                new_x = y
                new_y = x
                if distance_array[new_x][new_y] == 0:
                    distance_array[x][y] = random.randint(1, dist_range)
                else:
                    distance_array[x][y] = distance_array[new_x][new_y]
    return distance_array


def generate_cities(count):
    cities_array = [x for x in xrange(0, count)]
    shuffle(cities_array)
    cities_array.append(cities_array[0])
    return cities_array


def get_fitness(list, distance_list):
    fitness = 0
    for indx, value in enumerate(list):
        if indx == 0:
            fitness += distance_list[list[indx]][list[len(list) - 1]]
        else:
            fitness += distance_list[value][list[indx - 1]]
    return -fitness


def switch_random(cities_list):
    max_range = len(cities_list) - 2
    index_one = random.randint(0, max_range)
    index_two = random.randint(0, max_range)
    while index_one == index_two or index_two == 0:
        index_two = random.randint(0, max_range)
    number_one = cities[index_one]
    number_two = cities[index_two]
    if index_one == 0:
        cities_list[index_two] = number_one
        cities_list[index_one] = number_two
        cities_list[max_range + 1] = number_two
    else:
        cities_list[index_one] = number_two
        cities_list[index_two] = number_one
    print("switched " + "[" + str(index_one) + "]" + str(number_one) + " with [" + str(index_two) + "]" + str(number_two))


CITIES_COUNT = 100
DISTANCE_RANGE = 200
STARTTEMP = 1000
distance = init_distance_array(CITIES_COUNT, DISTANCE_RANGE)
cities = generate_cities(CITIES_COUNT)
initialCities = cities[:]
lastFitness = get_fitness(cities, distance)
print("Generated random distances")
print(cities)
numberOfRuns = 0
temp = float(STARTTEMP)
starttime = datetime.datetime.now()
epsilon = 0.001
while temp > epsilon:
    citiesCopy = cities[:]
    print("switching distances")
    switch_random(cities)
    newFitness = get_fitness(cities, distance)
    print("lastFitness:" + str(lastFitness) + " newFitness: " + str(newFitness) + " temp:" + str(temp))
    eFunctionResult = 0
    propability = random.uniform(0, 1)
    try:
        eFunctionResult = float(math.exp(float((newFitness - lastFitness)) / float(temp)))
    except OverflowError:
        eFunctionResult = float('inf')
    print("propability: " + str(propability) + " efunction result:" + str(eFunctionResult))

    if newFitness > lastFitness:
        lastFitness = newFitness
    elif propability < eFunctionResult:
        lastFitness = newFitness
    else:
        cities = citiesCopy
    print("finished run number")
    print(numberOfRuns)
    numberOfRuns += 1
    temp = temp - epsilon
    print("new temparature: " + str(temp) + str(epsilon))
    print(str(temp - epsilon))
print(str(datetime.datetime.now() - starttime) + " runtime")
print("initalDistance: " + str(-1 * get_fitness(initialCities, distance)) + " initial array: " + str(initialCities))
print("finalDistance: " + str(-1 * lastFitness) + " final array: " + str(cities))
print("number of runs: " + str(numberOfRuns))
