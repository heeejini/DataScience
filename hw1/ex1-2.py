def makeDict(K, V):
    dictionary = dict(zip(K,V))
    return dictionary

K = ('Korean', 'Mathematics', 'English')
V = (90.3, 85.5, 92.7)
D = makeDict(K, V)

#print dictionary result
print(D)
for k in K:
    print(k, D[k])

