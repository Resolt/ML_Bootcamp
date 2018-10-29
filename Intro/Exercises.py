print("7 to the power of 4 is: {}\n".format(pow(7, 4)))

print("Hi there Sam".split()); print('')

print("The diameter of {planet} is {km} kilometres!\n".format(planet="Earth", km=12742))

print([1, 2, [3, 4], [5, [100, 200, ['hello']], 23, 11], 1, 7][3][1][2][0]); print('')

d = {'k1': [1, 2, 3, {'tricky':['oh', 'man', 'inception', {'target':[1, 2, 3, 'hello']}]}]}
print(d['k1'][3]['tricky'][3]['target'][3]); print('')

def domainGet(txt):
	return txt.split('@')[1]

print(domainGet("test@domain.com")); print('')

def findDog(txt):
	return txt.find("Dog") != -1


if findDog("Where's my Dog?"):
	print("Yes!\n")


def countDog(txt):
	return txt.count("Dog")


print(countDog("Dog, Dog, Dog")); print('')

def filtNoS(lst):
	return filter(lambda txt:txt[0] == 's', lst)


seq = ['soup', 'dog', 'salad', 'cat', 'great']
print(filtNoS(seq)); print('')

def caughtSpeeding(speed, bday):
	if bday: speed -= 5
	
	if speed <= 60:
		print("No Ticket!\n")
	elif speed >= 61 and speed <= 80:
		print("Small Ticket!\n")
	else:
		print("Big Ticket!\n")
	

caughtSpeeding(81, False)
caughtSpeeding(81, True)
caughtSpeeding(65, True)

