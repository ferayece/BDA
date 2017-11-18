
# TASK-1: GENERAL
print('******** Case sensitivity *********')
# Case sensitive
array1 = 'help'
array2 = 'help'
array3 = 'Help'
print(array1 == array2)
print(array2 == array3)


Item_name = "Computer" #A String
Item_qty = 10 #An Integer
Item_value = 1000.23 #A floating point
print(Item_name)
# Computer
print(Item_qty)
# 10
print(Item_value)
# 1000.23

print('******** Index *********')
# Index starts from zero
print(array1)
print(array1[0])

# To get help
# help() # then please type modules
# help(str.replace)
# help(re)

# Whitespace for indentation
import os
import jupyter_console
# dir(os)

# SCALAR TYPES
# Integer
x = 1
print(x)
print(type(x))

# Float
y = 12.1
print(y)
print(type(y))

# Boolean: True or False
print(array1 == array2)
print(array2 == array3)

t = True
f = False
print(type(t))
print(type(f))

# Strings UNICODE in Python 3
print(type(array1))

# Special characters can be used in strings

string1 = 'dec%&d'
print(string1)

variable = None
if variable is None:
    print("Variable is None")
else:
    print("Variable is not None")

"""
\n	Newline
\t	Horizontal Tab
\\	Backslash
\'	Single Quote
\"	Double Quote"""


""" TUPLE
A tuple is a container which holds a series of comma-separated values (items or elements) between parentheses such as
an (x, y) co-ordinate. Tuples are like lists, except they are immutable (i.e. you cannot change its content once created)
and can hold mix data types. Tuples play a sort of "struct" role in Python -- a convenient way to pass around a little
logical, fixed size bundle of values."""

#create an empty tuple
tuplex = ()
print(tuplex)

# create a tuple with different data types
tuplex = ('tuple', False, 3.2, 1)
print (tuplex)
# ('tuple', False, 3.2, 1)

# create a tuple with numbers, notation without parenthesis
tuplex = 4, 7, 3, 8, 1
print (tuplex)
# (4, 7, 3, 8, 1)

# create a tuple of one item, notation without parenthesis
tuplex = 4,
print (tuplex)
# (4,)

# create an empty tuple with tuple() function built-in Python
tuplex = tuple()
print (tuplex)
# ()

# create a tuple from a iterable object
tuplex = tuple([True, False])
print (tuplex)
# (True, False)

#create a tuple
tuplex = ("w", 3, "r", "e", "s", "o", "u", "r", "c", "e")
print(tuplex)
#('w', 3, 'r', 'e', 's', 'o', 'u', 'r', 'c', 'e')

#get item (4th element)of the tuple by index
item = tuplex[3]
print(item)
# e
#get item (4th element from last)by index negative
item1 = tuplex[-4]
print(item1)
# u

#create a tuple
tuplex = ("w", 3, "r", "e", "s", "o", "u", "r", "c", "e")
print(tuplex)
# ('w', 3, 'r', 'e', 's', 'o', 'u', 'r', 'c', 'e')

#use in statements:
print("r" in tuplex)
# True
print(5 in tuplex)
# False


"""A list is a container which holds comma-separated values (items or elements) between square brackets where items or
elements need not all have the same type. In general, we can define a list as an object that contains multiple data items
(elements). The contents of a list can be changed during program execution. The size of a list can also change during
execution, as elements are added or removed from it.
Note: There are much programming languages which allow us to create arrays, which are objects similar to lists. Lists
serve the same purpose as arrays and have many more built-in capabilities. Traditional arrays can not be created in Python."""

numbers = [10, 20, 30, 40, 50]
names = ["Sara", "David", "Warner", "Sandy"]
student_info = ["Sara", 1, "Chemistry"]

my_list1 = [5, 12, 13, 14] # the list contains all integer values
print(my_list1)
#[5, 12, 13, 14]

my_list2 = ['red', 'blue', 'black', 'white'] # the list contains all string values
print(my_list2)
# ['red', 'blue', 'black', 'white']

my_list3 = ['red', 12, 112.12] # the list contains a string, an integer and a float values
print(my_list3)
# ['red', 12, 112.12]

color_list1 = ["White", "Yellow"]
color_list2 = ["Red", "Blue"]
color_list3 = ["Green", "Black"]
color_list = color_list1 + color_list2 + color_list3
print(color_list)
# ['White', 'Yellow', 'Red', 'Blue', 'Green', 'Black']

number = [1,2,3]
print(number[0]*4)
# 4

print(number*4)
# [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3]

color_list=["Red", "Blue", "Green", "Black"] # The list have four elements indices start at 0 and end at 3
print(color_list[0])  # Return the First Element
# 'Red'
print(color_list[0],color_list[3]) # Print First and Last Elements
# Red Black

print(color_list[-1])  # Return Last Element
# Black

#########################################################################################################
# ERROR print(color_list[4]) # Creates Error as the indices is out of range
# Traceback (most recent call last):

# Add an item to the end of the list:
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']

color_list.append("Yellow")
print(color_list)
# ['Red', 'Blue', 'Green', 'Black', 'Yellow']

# Insert an item at a given position:
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']

color_list.insert(2, "White") #Insert an item at third position
print(color_list)
# ['Red', 'Blue', 'White', 'Green', 'Black']

# Modify an element by using the index of the element:
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']

color_list[2]="Yellow"  #Change the third color
print(color_list)
# ['Red', 'Blue', 'Yellow', 'Black']

# Remove an item from the list:
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']

color_list.remove("Black")
print(color_list)
# ['Red', 'Blue', 'Green']

# Remove all items from the list:
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']
color_list.clear()
print(color_list)
# []

# List Slices
color_list=["Red", "Blue", "Green", "Black"] # The list have four elements
# indices start at 0 and end at 3
print(color_list[0:2]) # cut first two items
# ['Red', 'Blue']
print(color_list[1:2]) # cut second item
# ['Blue']
print(color_list[1:-2]) # cut second item
# ['Blue']
print(color_list[:3]) # cut first three items
# ['Red', 'Blue', 'Green']
print(color_list[:]) # Creates copy of original list
# ['Red', 'Blue', 'Green', 'Black']


# Remove the item at the given position in the list, and return it
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']
color_list.pop(2) # Remove second item and return it
# 'Green'
print(color_list)
# ['Red', 'Blue', 'Black']

# Return the index in the list of the first item whose value is x
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']
print(color_list.index("Red"))
# 0
print(color_list.index("Black"))
# 3

# Return the number of times 'x' appear in the list.
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']
color_list=["Red", "Blue", "Green", "Black", "Blue"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black', 'Blue']
color_list.count("Blue")
# 2

# Sort the items of the list in place:
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']

color_list.sort(key=None, reverse=False)
print(color_list)
# ['Black', 'Blue', 'Green', 'Red']

# Reverse the elements of the list in place:
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']
color_list.reverse()
print(color_list)
# ['Black', 'Green', 'Blue', 'Red']

# Search the Lists and find Elements:
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']
color_list.index("Green")
# 2

# Lists are Mutable:
# Items in the list are mutable i.e. after creating a list you can change any item in the list. See the following statements.
color_list=["Red", "Blue", "Green", "Black"]
print(color_list[0])
# Red
color_list[0]="White" # Change the value of first item "Red" to "White"
print(color_list)
# ['White', 'Blue', 'Green', 'Black']
print(color_list[0])
# White

# How to use the double colon [ : : ]?
listx=[1, 5, 7, 3, 2, 4, 6]
print(listx)
# [1, 5, 7, 3, 2, 4, 6]
sublist=listx[2:7:2] #list[start:stop:step], #step specify an increment between the elements to cut of the list.
print(sublist)
# [7, 2, 6]
sublist=listx[::3] #returns a list with a jump every 2 times.
print(sublist)
# [1, 3, 6]
sublist=listx[6:2:-2] #when step is negative the jump is made back
print(sublist)
# [6, 2]

# Find the largest and the smallest item in a list:
listx=[5, 10, 3, 25, 7, 4, 15]
print(listx)
# [5, 10, 3, 25, 7, 4, 15]
print(max(listx))	# the max() function of built-in allows to know the highest value in the list.
# 25
print(min(listx)) #the min() function of built-in allows to know the lowest value in the list.
# 3

# Compare two lists in Python:
listx1, listx2=[3, 5, 7, 9], [3, 5, 7, 9]
print (listx1 == listx2)
# True
listx1, listx2=[9, 7, 5, 3], [3, 5, 7, 9]	#create two lists equal, but unsorted.
print(listx1 == listx2)
# False
listx1, listx2 =[2, 3, 5, 7], [3, 5, 7, 9]	#create two different list
print(listx1 == listx2)
# False
print(listx1.sort() == listx2.sort())	#order and compare
# True

# Nested lists in Python:
listx = [["Hello", "World"], [0, 1, 2, 3, 4, 5]]
print(listx)
# [['Hello', 'World'], [0, 1, 2, 3, 4, 5]]
listx = [["Hello", "World"], [0, 1, 2, 3, 3, 5]]
print(listx)
# [['Hello', 'World'], [0, 1, 2, 3, 3, 5]]
print(listx[0][1])		#The first [] indicates the index of the outer list.
# World
print(listx[1][3])		#the second [] indicates the index nested lists.
# 3
listx.append([True, False])		#add new items
print(listx)
# [['Hello', 'World'], [0, 1, 2, 3, 3, 5], [True, False]]
listx[1][2]=4
print(listx)
# [['Hello', 'World'], [0, 1, 4, 3, 3, 5], [True, False]]		#update value items

# How can I get the index of an element contained in the list?
listy = list("HELLO WORLD")
print(listy)
# ['H', 'E', 'L', 'L', 'O', ' ', 'W', 'O', 'R', 'L', 'D']
index = listy.index("L")	#get index of the first item whose value is passed as parameter
print(index)
# 2
index = listy.index("L", 4)	#define the index from which you want to search
print(index)
# 9
index = listy.index("O", 3, 5)	#define the segment of the list to be searched
print(index)
# 4

# Using Lists as Stacks:
color_list=["Red", "Blue", "Green", "Black"]
print(color_list)
# ['Red', 'Blue', 'Green', 'Black']
color_list.append("White")
color_list.append("Yellow")
print(color_list)
# ['Red', 'Blue', 'Green', 'Black', 'White', 'Yellow']
color_list.pop()
# 'Yellow'
color_list.pop()
# 'White'
color_list.pop()
# 'Black'
color_list
# ['Red', 'Blue', 'Green']


# Swap variables
# Python swap values in a single line and this applies to all objects in python.
# Syntax: var1, var2 = var2, var1
# Example:
x1 = 10
y1 = 20
print(x1)
# 10
print(y1)
# 20
x1, y1 = y1, x1
print(x1)
# 20
print(y1)
# 10


"""Dictionary
Python dictionary is a container of the unordered set of objects like lists. The objects are surrounded by curly braces
{ }. The items in a dictionary are a comma-separated list of key:value pairs where keys and values are Python data type.

Each object or value accessed by key and keys are unique in the dictionary. As keys are used for indexing, they must be
the immutable type (string, number, or tuple). You can create an empty dictionary using empty curly braces."""


# Create a new dictionary in Python
#Empty dictionary
new_dict = dict()
new_dict = {}
print(new_dict)
# {}

# Dictionary with key-value
color = {"col1" : "Red", "col2" : "Green", "col3" : "Orange" }
print(color)
# {'col2': 'Green', 'col3': 'Orange', 'col1': 'Red'}

# Get value by key in Python dictionary
#Declaring a dictionary
dict = {1:20.5, 2:3.03, 3:23.22, 4:33.12}
#Access value using key
print(dict[1])
# 20.5
print(dict[3])
# 23.22

#Accessing value using get() method
dict.get(1)
# 20.5
dict.get(3)
# 23.22

# Add key/value to a dictionary in Python
#Declaring a dictionary with a single element
dic = {'pdy1':'DICTIONARY'}
print(dic)
# {'pdy1': 'DICTIONARY'}
dic['pdy2'] = 'STRING'
print(dic)
# {'pdy1': 'DICTIONARY', 'pdy2': 'STRING'}

#Using update() method to add key-values pairs in to dictionary
d = {0:10, 1:20}
print(d)
# {0: 10, 1: 20}
d.update({2:30})
print(d)
# {0: 10, 1: 20, 2: 30}

# Remove a key from a Python dictionary
# Code:
myDict = {'a':1,'b':2,'c':3,'d':4}
print(myDict)
if 'a' in myDict:
    del myDict['a']
print(myDict)
# Output:
# {'d': 4, 'a': 1, 'b': 2, 'c': 3}
# {'d': 4, 'b': 2, 'c': 3}

# Sort a Python dictionary by key
# Code:
color_dict = {'red':'#FF0000',
          'green':'#008000',
          'black':'#000000',
          'white':'#FFFFFF'}

for key in sorted(color_dict):
    print("%s: %s" % (key, color_dict[key]))
# Output:
# black: #000000
# green: #008000
# red: #FF0000
# white: #FFFFFF


# Find the maximum and minimum value of a Python dictionary
# Code:

my_dict = {'x':500, 'y':5874, 'z': 560}
key_max = max(my_dict.keys(), key=(lambda k: my_dict[k]))
key_min = min(my_dict.keys(), key=(lambda k: my_dict[k]))
print('Maximum Value: ',my_dict[key_max])
print('Minimum Value: ',my_dict[key_min])
# Output:
# Maximum Value:  5874
# Minimum Value:  500

# Concatenate two Python dictionaries into a new one
# Code:
dic1={1:10, 2:20}
dic2={3:30, 4:40}
dic3={5:50,6:60}
dic4 = {}
for d in (dic1, dic2, dic3): dic4.update(d)
print(dic4)
# Output:
# {1: 10, 2: 20, 3: 30, 4: 40, 5: 50, 6: 60}

# Test whether a Python dictionary contains a specific key
# Code:
fruits = {}
fruits["apple"] = 1
fruits["mango"] = 2
fruits["banana"] = 4

# Use in.
if "mango" in fruits:
    print("Has mango")
else:
    print("No mango")

# Use in on nonexistent key.
if "orange" in fruits:
    print("Has orange")
else:
    print("No orange")
# Output
# Has mango
# No orange
# Find the length of a Python dictionary
# Code:
fruits = {"mango": 2, "orange": 6}

# Use len() function to get the length of the dictionary
print("Length:", len(fruits))
# Output:
# Length: 2

""" Sets
A set object is an unordered collection of distinct hashable objects. It is commonly used in membership testing,
removing duplicates from a sequence, and computing mathematical operations such as intersection, union, difference,
and symmetric difference.

Sets support x in the set, len(set), and for x in set like other collections. Set is an unordered collection and does
not record element position or order of insertion. Sets do not support indexing, slicing, or other sequence-like behavior.

There are currently two built-in set types, set, and frozenset. The set type is mutable - the contents can be changed
using methods like add() and remove(). Since it is mutable, it has no hash value and cannot be used as either a
dictionary key or as an element of another set. The frozenset type is immutable and hashable - its contents cannot be
altered after it is created; it can, therefore, be used as a dictionary key or as an element of another set.

Iteration Over Sets:
We can move over each of the items in a set using a loop. However, since sets are unordered, it is undefined which order
the iteration will follow."""

num_set = set([0, 1, 2, 3, 4, 5])
for n in num_set:
  print(n)
# 0 1 2 3 4 5

# Add member(s) in Python set:
# A new empty set
color_set = set()

# Add a single member
color_set.add("Red")
print(color_set)
# {'Red'}

# Add multiple items
color_set.update(["Blue", "Green"])
print(color_set)
# {'Red', 'Blue', 'Green'}

# Remove item(s) from Python set:
# pop(), remove() and discard() functions are used to remove individual item from a Python set.
# pop() function:

num_set = set([0, 1, 2, 3, 4, 5])
num_set.pop()
# 0
print(num_set)
# {1, 2, 3, 4, 5}
num_set.pop()
# 1
print(num_set)
# {2, 3, 4, 5}

# remove() function:
num_set = set([0, 1, 2, 3, 4, 5])
num_set.remove(0)
print(num_set)
# {1, 2, 3, 4, 5}

# discard() function:
num_set = set([0, 1, 2, 3, 4, 5])
num_set.discard(3)
print(num_set)
# {0, 1, 2, 4, 5}

# Intersection of sets:
# In mathematics, the intersection A ∩ B of two sets A and B is the set that contains all elements of A that also
# belong to B (or equivalently, all elements of B that also belong to A), but no other elements.
# Intersection
setx = set(["green", "blue"])
sety = set(["blue", "yellow"])
setz = setx & sety
print(setz)
# {'blue'}

# Union of sets:
# In set theory, the union (denoted by ∪) of a collection of sets is the set of all distinct elements in the collection.
# It is one of the fundamental operations through which sets can be combined and related to each other.
# Union
setx = set(["green", "blue"])
sety = set(["blue", "yellow"])
seta = setx | sety
print (seta)
# {'yellow', 'blue', 'green'}

# Set difference:
setx = set(["green", "blue"])
sety = set(["blue", "yellow"])
setz = setx & sety
print(setz)
# {'blue'}

# Set difference
setb = setx - setz
print(setb)
# {'green'}

# Symmetric difference:
setx = set(["green", "blue"])
sety = set(["blue", "yellow"])

#Symmetric difference
setc = setx ^ sety
print(setc)
# {'yellow', 'green'}

# issubset and issuperset:
setx = set(["green", "blue"])
sety = set(["blue", "yellow"])
issubset = setx <= sety
print(issubset)
# False

issuperset = setx >= sety
print(issuperset)
# False

# More examples:
setx = set(["green", "blue"])
sety = set(["blue", "green"])
issubset = setx <= sety
print(issubset)
# True

issuperset = setx >= sety
print(issuperset)
# True

# Shallow copy of sets:
setx = set(["green", "blue"])
sety = set(["blue", "green"])
# A shallow copy
setd = setx.copy()
print(setd)
# {'blue', 'green'}

# Clear sets:
setx = set(["green", "blue"])
#Clear AKA empty AKA erase
sete = setx.copy()
sete.clear()
print(sete)
# set()

# Local and global variables in Python
# In Python, variables that are only referenced inside a function are implicitly global. If a variable is assigned
# a value anywhere within the function’s body, it’s assumed to be a local unless explicitly declared as global.
# Example:
var1 = "Python"
def func1():
    var1 = "PHP"
    print("In side func1() var1 = ",var1)

def func2():
    print("In side func2() var1 = ",var1)

func1()
func2()
# Output:
# In side func1() var1 =  PHP
# In side func2() var1 =  Python
# You can use a global variable in other functions by declaring it as global keyword :

# Example:
def func1():
    global var1
    var1 = "PHP"
    print("In side func1() var1 = ",var1)

def func2():
    print("In side func2() var1 = ",var1)

func1()
func2()
# Output:
# In side func1() var1 =  PHP
# In side func2() var1 =  PHP

