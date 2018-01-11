"""
Evolution is a evolutionary algorithm which aim is to evolve a matching bit string
Given is a population P, from which we select the best individuals and add them with a given ration of
randomly chosen and mated individuals


"""
import random
import datetime


POPULATION_SIZE = 2000
LENGTH_OF_INDIVIDUAL = 1000

# r is the portion of the population which will be replaced by a crossover
r = 0.3
# m is the mutation rate
m = 0.3


def gen_rand_bitstring(length):
    bitstring = ""
    for i in xrange(0, length):
        bitstring += "1" if random.uniform(0, 1) > 0.5 else "0"
    return bitstring


class Individual:
    def __init__(self, information, comparator):
        self.information = information
        self.fitness = self.compute_fitness(comparator)

    def compute_fitness(self, comparator):
        distance = 0
        for idx, char in enumerate(comparator):
            if char != self.information[idx]:
                distance -= 1

        return distance

    def crossover(self, other, comparator):
        new_string = ""
        # use one random crossover point instead on each single crossover
        for idx, char in enumerate(self.information):
            new_string += self.information[idx] if random.uniform(0, 1) > 0.5 else other.information[idx]
        return Individual(new_string, comparator)

    def mutate(self, comparator):
        # change random bit
        rand_index = random.randint(0, len(self.information) - 1)
        mutation = self.information[0: rand_index]
        mutation += "1" if self.information[rand_index] == "0" else "0"
        if rand_index + 1 < len(self.information) and rand_index + 1 <= len(self.information):
            mutation += self.information[rand_index + 1: len(self.information)]
        self.fitness = self.compute_fitness(comparator)


class Population:
    def __init__(self, size, length, comparator, mutation_rate, crossover_ratio, max_runs):
        new_population = list()
        for i in xrange(0, size):
            new_population.append(Individual(gen_rand_bitstring(length), comparator))
        self.population = new_population
        self.m = mutation_rate
        self.r = crossover_ratio
        self.comparator = comparator
        self.max_runs = max_runs

    def get_best_individual(self):
        best_individual = self.population[0]
        for idx, individual in enumerate(self.population):
            if individual.fitness > best_individual.fitness:
                best_individual = individual
        return best_individual

    def crossover(self):
        crossover_size = int(self.r * len(self.population)/2)
        print('performing crossover of ' + str(crossover_size) + ' individuals ...')
        crossed = list()
        chosen = list()
        for i in xrange(0, crossover_size):
            indx1 = random.randint(0, len(self.population) - 1)
            indx2 = random.randint(0, len(self.population) - 1)
            while indx1 == indx2 or indx1 in chosen or indx2 in chosen:
                indx1 = random.randint(0, len(self.population) - 1)
                indx2 = random.randint(0, len(self.population) - 1)
            chosen.append(indx1)
            chosen.append(indx2)
            crossed.append(self.population[indx1].crossover(self.population[indx2], self.comparator))
            crossed.append(self.population[indx2].crossover(self.population[indx1], self.comparator))
        print('crossover done!')
        return crossed

    def selection_rank(self):
        rank_list_size = int((1 - self.r) * len(self.population))
        rank_list_size += 0 if rank_list_size % 2 == 0 else rank_list_size + 1
        print('rank selecting ' + str(rank_list_size) + ' individuals')
        selected = list()
        selected.extend(sorted(self.population, key=lambda individual: -individual.fitness)[0:rank_list_size])
        return selected

    def one_evolution_step(self):
        new_population = list()
        new_population.extend(self.selection_rank())
        new_population.extend(self.crossover())
        mutated = 0
        for idx, indv in enumerate(new_population):
            if random.uniform(0, 1) <= self.m:
                new_population[idx].mutate(self.comparator)
                mutated += 1
        print('mutated ' + str(mutated) + ' individuals')
        self.population = new_population

    def evolve(self):
        number_of_runs = 0
        start_time = datetime.datetime.now()
        best_individual = self.get_best_individual()
        while best_individual.fitness < 0 and number_of_runs < self.max_runs:
            round_time = datetime.datetime.now()
            number_of_runs += 1
            print('#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.#.# run number:',str(number_of_runs))
            self.one_evolution_step()
            best_individual = self.get_best_individual()
            print('best string ' + str(best_individual.fitness) +':' + best_individual.information)
            print('searched string :' + self.comparator)

            print('round runtime: ' + str(datetime.datetime.now()-round_time))
        print('found individual with fitness of 0 : ' + best_individual.information)
        print('runs: ' + str(number_of_runs))
        return datetime.datetime.now() - start_time


random_bit_string = gen_rand_bitstring(LENGTH_OF_INDIVIDUAL)
print('generated random string size ' + str(LENGTH_OF_INDIVIDUAL) + ' : ' + random_bit_string)
population = Population(POPULATION_SIZE, LENGTH_OF_INDIVIDUAL, random_bit_string, m, r, 1000)
runtime = population.evolve()
print('runtime: ' + str(runtime) + 's')
