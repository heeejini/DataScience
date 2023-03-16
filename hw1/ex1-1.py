import math

def isprime(num):
    for i in range(2, int(math.sqrt(num)) + 1):  # the number from 2 to the square root of x
        if num % i == 0:  # A number that is divisible is not prime
            print(num, "is not Prime, divisible by", i)
            return False
    return True

cnt =0
for i in range(2, 32768):
    if isprime(i)==True:
        print(i, "is Prime")
        cnt+=1

print("total prime count : ",cnt)