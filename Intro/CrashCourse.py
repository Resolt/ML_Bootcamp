# FORMAT STRINGS
print('My age is {one} and my name is {two}'.format(one=pow(2, 5), two="LEL"))
three = 200
four = "DINGLEDONG"
print('My age is {} and my name is {}'.format(three, four))

# INDEXING STRINGS
alph = 'abcdefghijkl'
print(alph[:4])
print(alph[6:8])
print(alph[2])
print(alph[2:3])
print(alph[2:4])

# LISTS
print(['a', 'b', 'c'])
my_list = ['a', 'b', 'c', 'd', 'e', 'f']
my_list.append('W')
print(my_list)
print(my_list[2])
print(my_list[2:3])
print(my_list[2:4])

nest = [1, [2, 3]]
print(nest)
print(nest[0])
print(nest[1][0])
print(nest[1][1])
nest2 = [1, 2, 3, 4, [5, 6, 7, ['target']]]
print(nest2)
print(nest2[4][3][:])
print(nest2[4][3][0])

# DICTIONARIES
d = {'key1': 'Key1Value', 'key2': 4050}
print(d)
print(d['key1'])
nestedDict = {'key1': {'nkey1': [1, 2, 3]}}
print(nestedDict['key1']['nkey1'])
print(nestedDict['key1']['nkey1'][1])

# TUPLES
t = (1, 2, 3, 4)
print(t[0:2])
#t[3] = 5 # THIS WON'T WORK AS TUPLES CAN'T BE MODIFIED - THEY ARE const

# SETS
print({1, 2, 3})
print({1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3})
s = {1, 1, 1, 1, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4}
s.add(5)
print(s)

# IF STATEMENTS
if 1 < 2:
	print("Aww yearh")

if False:
	print("BAWLS")

if True:
	print("Yes")
elif 2 > 1:
	print("Also yes")
else:
	print("BAWLS")

if True and False:
	print("Yes and yes")
elif True or False:
	print("Yes and/or no")

# LOOPS
for i in range(5):
	print(i)
	
i = 0
while i < 5 or False:
	print(i)
	i += 1
	
out1 = []
for num in range(5):
	out1.append(num**2)

print(out1)

# LIST COMPREHENSION
out2 = [num ** 2 for num in range(5)]
print(out2)

# FUNCTIONS
def func(a, b):
	"""
	This is a docstring.
	Multiple line comment for documentation.
	Trolololololo.
	"""
	return a + b

print(func(5, 6))

tot = ''

for l in ['a', 'b', 'c', 'd', 'e']:
	tot = func(tot, l)

print(tot)

# MAP FUNCTION
seq = list(range(5))

def func2(a):
	return a * 2

print(map(func2, seq))
print(map(func, seq, seq))

# LAMBDA EXPRESSIONS (ANONYMOUS FUNCTIONS)
print(map(lambda var: var + 1, seq))
print([var + 1 for var in seq])  # THIS GIVES THE SAME RESULT, BUT IT'S MORE DIFFICULT TO DEAL WITH MULTIPLE ARGUMENTS

# FILTER FUNCTION
print(filter(lambda n: n % 2 == 0, seq))

# METHODS
s = 'Hello, my name is Jeff'
t = s.split()
print(t)
print(s.split(', '))
print(len(s))
print(len(t))

d = {'k1': 1, 'k2': 2}
print(d.keys())
print(d.values())

lst = [1, 2, 3, 4, 5]
lst.pop()
print(lst)
lst.extend(lst)
print(lst)
lst2 = []
while len(lst) > 0:
	lst2.append(lst.pop(len(lst) - 1))

print(lst)
print(lst2)
lst2.reverse()
print(lst2)

print('x' in ['a', 'b', 'c'])
print(1 in [1, 2, 3])

if 'x' in ['x', 'y', 'z']:
	print("YAS")

# TUPLE UNPACKING

y = [(1, 2), (3, 4), (5, 6)]
for a, b in y:
	print(a)

d = y[0]
print(d)

d, f = y[0]
print(str(d)+' '+str(f))

