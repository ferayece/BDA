# A new empty set
setx = set()
print(setx)
#set()
#A non empty set
n = set([0, 1, 2, 3, 4, 5])
print(n)
# {0, 1, 2, 3, 4, 5}


"""
def function_name(argument1, argument2, ...) :
    statement_1
    statement_2
    ....
"""

def avg_number(x, y):
    print("Average of ",x," and ",y, " is ",(x+y)/2)
avg_number(3, 4)


"""
TASK-1: Please write a multiplication function that takes two integers and outputs
the product. For instance the output should be as follows for the following x=2, y=3:
2 * 3 = 6
"""

def multiplication(x, y):
    print("Multiplication of ",x," and ",y, " is ",x*y)
multiplication(2,3)


"""if expression :
    statement_1
    statement_2
    ...."""

x = 1
if (x > 0):
    print("Greater than zero!")


"""if expression :
    statement_1
    statement_2
    ....
else :
   statement_3
   statement_4
   ...."""


a=10
if(a>10):
    print("Value of a is greater than 10")
else:
    print("Value of a is 10")


"""
if expression1 :
    statement_1
    statement_2
    ....
elif expression2 :
     statement_3
     statement_4
     ....
elif expression3 :
     statement_5
     statement_6
     ....................
else:
     statement_7
     statement_8
"""

#create two boolean objects
x = False
y = True
#The validation will be True only if all the expressions generate a value True
if x and y:
    print('Both x and y are True')
else:
    print('x is False or y is False or both x and y are False')

#Task 1.5:

x = True
y = False
#The validation will be True only if all the expressions generate a value True
if x and y:
    print('Both x and y are True')
elif x:
    print('Just X is True.')
elif y:
    print('Just Y is True.')
else:
    print('Both x and y are False')
    #print('x is False or y is False or b


"""
TASK-2:
Please write a function that takes an integer and gives the output in relation to the number positive or negative
and odd or even. Here is the example: x = 74
The given number, 74, is positive and even.
"""

def Explain_Num(x):
    if(x>0):
        if(x%2==1):
            print("Given number ", x, "is positive and odd.")
                # print("Multiplication of ", x, " and ", y, " is ", x * y)
        else:
            print("Given number, ", x, ",is positive and even.")
    else:
        if(x%2==1):
            print("Given number, ", x, ",is negative and odd.")
                # print("Multiplication of ", x, " and ", y, " is ", x * y)
        else:
            print("Given number, ", x, ",is negative and even.")
Explain_Num(-1)
Explain_Num(3)
Explain_Num(74)
Explain_Num(75)
Explain_Num(0)

# Create a integer
n = 150
print(n)
# If n is greater than 500, n is multiplied by 7, otherwise n is divided by 7
result = n * 7 if n > 500 else n / 7
print(result)



"""
for variable_name in sequence :
    statement_1
    statement_2
    ....
"""

#The list has four elements, indices start at 0 and end at 3
color_list = ["Red", "Blue", "Green", "Black"]
for c in color_list:
    print(c)


"""for <variable> in range(<number>):"""

for a in range(4):
  print(a)

"""for "variable" in range("start_number", "end_number"):"""

for a in range(2,7):
    print(a)

"""
for "variable" in range("start_number",  "end_number", "increment_number"):
range(a,b,c): Generates a sequence of numbers from a to b excluding b, incrementing by c.
"""

for a in range(2, 7, 2):
    print(a)


"""Example: Iterating over tuple"""

numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9) # Declaring the tuple
count_odd = 0
count_even = 0
for x in numbers:
    if x % 2:
    	count_odd+=1
    else:
    	count_even+=1
print("Number of even numbers :",count_even)
print("Number of odd numbers :",count_odd)


"""Example: Iterating over dictionary
In the following example for loop iterates through the dictionary "color" through its keys and prints each key."""

color = {"c1": "Red", "c2": "Green", "c3": "Orange"}
for key in color:
   print(key)

color = {"c1": "Red", "c2": "Green", "c3": "Orange"}
for key in color:
   print(key,color.get(key))

for key.value in color.items():
    print (key,value)

color = {"c1": "Red", "c2": "Green", "c3": "Orange"}
for key,value in color.items():
    print(key,value)

for value in color.values():
    print(value)



"""Following for loop iterates through its values :"""

color = {"c1": "Red", "c2": "Green", "c3": "Orange"}
for value in color.values():
   print(value)


"""
Extra:
for variable_name in sequence :
    statement_1
    statement_2
    ....
else :
    statement_3
    statement_4
    ....
The else clause is only executed after completing the for loop. If a break statement executes in first program block and terminates the loop then the else clause does not execute.
"""

"""
Write a Python program to find those numbers which are divisible by 7 and multiple of 5, between 1500 and 2700 (both included).
"""

nl = []
for x in range(1500, 2700):
    if (x % 7 == 0) and (x % 5 == 0):
        nl.append(str(x))
print(','.join(nl))

"""
TASK-3:
Write a Python program to find the numbers which are divisible by 131 or multiple of 53, between 100 and 2700 (both included).
Print the numbers on the screen with a space between them
"""
nl = []
for x in range(100, 2701):
    if (x % 131 == 0) or (x % 53 == 0):
        nl.append(str(x))
print(' '.join(nl))


"""while (expression) :
    statement_1
    statement_2
    ...."""

x = 0
while (x < 5):
     print(x)
     x += 1

"""
The sum of first 9 integers
"""
x = 0
s = 0
while (x < 10):
     s = s + x
     x = x + 1
else:
     print('The sum of first 9 integers : ',s)


"""
Please try to understand the code flow and guess the output
"""
x = 1
s = 0
while (x < 10):
     s = s + x
     x = x + 1
     if (x == 5):
          break
else :
     print('The sum of first 9 integers : ',s)
print('The sum of ',x,' numbers is :',s)


"""
Write a Python program to get the Fibonacci series between 0 to 50.
Note : The Fibonacci Sequence is the series of numbers :
0, 1, 1, 2, 3, 5, 8, 13, 21, ....
Every next number is found by adding up the two numbers before it.
"""

x, y = 0, 1
while y < 50:
    print(y)
    x, y = y, x+y

"""
TASK-4:
Write a Python program to get the Tibonacci series between 0 to 50.
The Tibonacci Sequence is the series of numbers :
0, 1, 1, 2, 4, 7, 13, 24, 44, ....
Every next number is found by adding up the three numbers before it.
"""
x, y,z = 0,1,1
while x < 50:
    print(x)
    x,y,z=y,z,x+y+z
"""
while (expression1) :
     statement_1
     statement_2
     ......
     if expression2 :
        break
for variable_name in sequence :
   statement_1
   statement_2
   if expression3 :
        break
"""

numbers = (1, 2, 3, 4, 5, 6, 7, 8, 9) # Declaring the tuple
num_sum = 0
count = 0
for x in numbers:
    num_sum = num_sum + x
    count = count + 1
    if count == 5:
        break
print("Sum of first ",count,"integers is: ", num_sum)



# factorial.py
def factcal(n): # Create the factorial of a positive integer
    fact = 1
    while n>0:
          fact *= n
          n=n-1
          if(n<=1):
            break
    else: # Display the message if n is not a positive integer.
          print('Input a correct number....')
          return
    return fact

def factdata(n): # return the numbers of factorial x
    result = []
    while n>0:
       result.append(n)
       n = n - 1
       if(n==0):
        break
    else: # Display the message if n is not a positive integer.
       print('Input a correct number....')
       return
    return result

"""
TASK-5:
Write a Python program to check a triangle is equilateral, isosceles or scalene. Note:
An equilateral triangle is a triangle in which all three sides are equal.
A scalene triangle is a triangle that has three unequal sides.
An isosceles triangle is a triangle with (at least) two equal sides.
"""

def check_triangle(a,b,c):
    if a==b:
        if b==c:
            print('An equilateral - eskenar triangle!')
        else:
            print('A isosceles -ikizkenar triangle!')
    elif b==c:
        print('A isosceles ikizkenar triangle!')
    elif a==c:
        print('A isosceles ikizkenar triangle!')
    else:
        print('An scalene-çeşitkenar triangle')

check_triangle(3,3,3)
check_triangle(3,2,3)
check_triangle(1,2,3)
check_triangle(2,3,3)
check_triangle(3,2,3)
check_triangle(3,3,2)
check_triangle(2,3,6)
check_triangle(3,2,2)
check_triangle(5,5,5)

def check_triangle(a,b,c):
    if a==b and b==c:
            print('An equilateral - eskenar triangle!')
    elif b==c:
            print('A isosceles -ikizkenar triangle!')
    elif a==c:
        print('A isosceles ikizkenar triangle!')
    else:
        print('An scalene-çeşitkenar triangle')


check_triangle(3,3,3)
check_triangle(3,2,3)
check_triangle(1,2,3)
check_triangle(2,3,3)
check_triangle(3,2,3)

check_triangle(2,3,6)
check_triangle(3,2,2)
check_triangle(5,5,5)




"""
Write a Python program to check the validity of a password (input from users).
Validation:
At least 1 letter between [a-z] and 1 letter between [A-Z].
At least 1 number between [0-9].
At least 1 character from [$#@].
Minimum length 6 characters.
Maximum length 16 characters.
"""
import re
p = input("Input your password")
x = True
while x:
    if (len(p)<6 or len(p)>12):
        break
    elif not re.search("[a-z]",p):
        break
    elif not re.search("[0-9]",p):
        break
    elif not re.search("[A-Z]",p):
        break
    elif not re.search("[$#@]",p):
        break
    elif re.search("\s",p):
        break
    else:
        print("Valid Password")
        x = False
        break
if x:
    print("Not a Valid Password")



