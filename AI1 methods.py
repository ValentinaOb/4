import random

# Параметри
POP_SIZE = 10
GENES = 5  # 5 бітів -> числа від 0 до 31
GENERATIONS = 3
MUTATION_RATE = 0.1

# Фітнес-функція: максимум x^2
def fitness(individual):
    x = int("".join(str(bit) for bit in individual), 2)
    print(individual, '  x ', x**2)
    return x**2

# Створення випадкової особини
def create_individual():
    indiv=[random.randint(0, 1) for _ in range(GENES)]
    print('Indiv ',indiv)
    return indiv

# Схрещування двох особин
def crossover(parent1, parent2):
    point = random.randint(1, GENES - 1)
    print('point ',point, 'Child', parent1[:point] + parent2[point:])
    return parent1[:point] + parent2[point:]

# Мутація
def mutate(individual):
    bits=[]
    for bit in individual:
        rand= random.random()
        
        print('rand',rand, bit, rand> MUTATION_RATE )
        if  rand<= MUTATION_RATE:
            bit=1 - bit
        bits.append(bit)
    return bits
#[
#        bit if random.random() > MUTATION_RATE else 1 - bit
#        for bit in individual]


# Генетичний алгоритм
population = [create_individual() for _ in range(POP_SIZE)]

for generation in range(GENERATIONS):
    # Оцінювання
    scored = [(fitness(ind), ind) for ind in population]
    scored.sort(reverse=True)
    population = [ind for _, ind in scored]
    print('population ',population)

    print(f"Gen {generation}: Best fitness = {fitness(population[0])}")

    # Відбір найкращих
    next_generation = population[:2]  # elitism: залишаємо 2 найкращих
    print('next_generation ',next_generation)

    # Створення решти нових особин
    while len(next_generation) < POP_SIZE:
        parent1 = random.choice(population[:5])
        parent2 = random.choice(population[:5])
        print('parent1 ',parent1)
        print('parent2 ',parent2)
        
        child = crossover(parent1, parent2)
        child = mutate(child)
        print('After Mutation', child)
        next_generation.append(child)

    population = next_generation
    print('new population', population)
