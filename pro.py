import cProfile

def func(a):
    sum = 0
    for i in range(a):
        sum += i
    return sum 


if  __name__ == "__main__":
    cProfile.run("func(10000000)")