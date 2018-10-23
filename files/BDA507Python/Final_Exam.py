print("BDA 507 SOLUTIONS OF FINAL EXAM, FERAY ECE TOPCU")
print("******************Beginning of Part A*******************")
# PART A (30%): The main objective of this part is to assess the learner’s main understanding of Python as a programming language
# with respect to other programming languages. Each question is worth 10 points for a total of 30 points.
# Every term like “high-level language” needs further explanation not to lose any points.

# Task-1: Why is Python, as a programming language, highly preferred by data scientists, data analists and AI researchers.
# Please provide appropriate (2 to 3) examples. When compared to other programming languages,
# what are the advantages and disadvantages of Python programming language? (150-250 words required).
print("********************Part A - Task 1************************")
print("1.It has a lot of library to manipulate, play, analyze and visualize the data.Python s popularity for data science is largely due to the strength 7"
      "of its core libraries (NumPy, SciPy, pandas,scikit-learn,seaborn ,matplotlib, IPython),"
      "high productivity for prototyping and building small and reusable systems, and its strength as a general purpose programming language.\n")
print("2.Thanks to large and free library enviroment; dealing"
  "-with manipulation of data,"
   "-with N-dimensional arrays, matrices,"
  " -with linear algebra, Fourier transform, and random number and etc."
   "-with visualizing data with fancy graphs, tables"
"is much more easier when we compared with C/C++, Java.")
print("3. The syntax is user friendly, consistent and elegant when we compare to R, Java, C/C++.Additionally, Python emphasizes productivity and readability."
   "It has easy syntax (close to real life grammer). Because of it, it is easy to learn. So, you can build a program very quickly with Python.\n"
"4. You can write object-oriented and functional code.\n"
"5. It s free to use. (open source language)\n"
"6. It has large community who used it, so finding soultions with python is easier by communicating with other people.\n"
"   Therefore, there are a lot of resources to learn and use Python nowadays.\n"
"7. Production phase. Python is a programming language which is designed for produces usable programs in organizations while R is a statistical programming.\n"
"   It means Python is designed for specifically generate programs. However, with R this is more harder.\n"
 "   R can be used like Excel, or another data analysis tool, It has wide range of uses.")

# Task-2: What are the main differences between Python 2 and Python 3? We had a slight mention
# for this by the beginning of the term but it is important to have a solid background about this.
# Why/why not Python 3? (150-250 words required).
print("********************Part A - Task 2************************")
print("1. The difference between print function: In Python 2, print acts like a statement rather than a function.\n"
      "There is no paratheses and this can be confusing because the other functions of Python 2 needs paranthesis and some arguments inside them.\n"
      "Also, without paranthesis it can give unexpected outputs. However Python 3 explicitly treats print as a function which means\n"
      "you can pass the items in parantheses. Paranthesis can help prevent mistakes.")
print("2. The difference between Unicode Strings: Python 3 stores strings as Unicode by default while Python 2 requires to mark a string as Unicode with\n"
      " 'u'. It is important for example when we try to make text analysis in different languages, so you don't have to label your text as Unicode in Python 3.")
print("3. The difference between Raising Exceptions: Python 3 requires different syntax for raising exceptions.")
print("4. The difference between Integer devision:The division of two integers returns a float instead of an integer. '//' can be used to have the 'old' behaviour.\n"
      "5. The difference between Integer Type: There is only one integer type left, i.e. int. long is int as well.")

# Task-3: Please provide a very brief history of Python as a programming language (150-250 words).
print("********************Part A - Task 3************************")

print("Python is an interpreted high-level programming language for programming.\n "
      "High-level language refers to the higher level of abstraction from machine language.\n"
      "High-level programming language means enables a programmer to write programs \n"
      "that are more or less independent of a particular type of computer. Such languages are considered high-level\n "
      "because they are closer to human languages and further from machine languages. What does it mean 'interpreted?' It means\n"
      "python is a type of programming language can execute its programms without previously compiling a program into machine-language instructions.\n"
      "PYthon has a interpreter which executes the program directly.\n"
      "Python is created by Guido van Rossum, he started implementation in the  late 1980s, and first released in 1991. . This release included already \n"
      "exception handling, functions, and the core data types of list, dict, str and others. It was also object oriented and had a module system. \n"
      " It has a vision as emphasizes code readability, easibility to learn. \n"
      "Python 2.0 was released on 16 October 2000 and had many major new features, including a cycle-detecting garbage collector and support for Unicode. \n"
      "Python 3.0 was released in 2008 with extended features. Python 3 is not backwards compatible with Python 2.x. \n"
      " During this days, Python is utilized by the likes of Nokia, Google, and even NASA. \n")


print("******************Beginning of Part B*******************")
# PART B (30%): To be able to write very basic algorithms with an awareness of Python syntax.
#
# Task-1: A friend of you wants to see the odd integers between 101 and 909.
# Thus, she expects from you to write a script (mainly composed of a loop).
# Then please extend it to a function that will take 3 integers from the user. For instance 100, 5, 130.
# Then it should print the following as an output: 100, 105, 110, 115, 120, 125, 130. (Any format for demonstration is accepted, here, the content is relevant).
# Then please put these integers into a list and thus be able to store them. (Hint: You can use the append function here).
# Finally, get input from the user (three integers). Please ensure that these are all integers.
# If not give an appropriate warning (You can use assert function).
print("********************Part B - Task 1************************")

# 1 Manuel print odd_numbers:
print("Part B Task 1: Manuel print odd numbers")
for i in range(100,909+1):
    if(i%2!=0):  ## is it odd?
        print(i)
print("Part B Task 1: End of manuel print odd numbers")

#Then extend it to a function (dynamic numbers)
def print_num(x,y,z):
    numlist=[]       ##create an empty list.
    for i in range(x,z+1,y): #borders are included. So, second one should be +1. range of the loop: from x to z, y by y.
        numlist.append(i)    #append x + y into list.
    return numlist

#try the function:
#take integer value from user:
print("Part B Task 1: do it with function: ")
x = int(input("Please enter an integer value for the lower border:"))
y = int(input("Please enter an integer value for the upper border:"))
z = int(input("Please enter an integer value for the increment:"))
print(print_num(x,z,y))   #print the list which is generated in the class instance.

print("***** End of Part B Task 1  ****")
############################################################################################
#Task-2: The same friend of yours asks you about the use of break, continue, and assert.
#Please provide her a good example to show their use and comparison with your suffienct explanations about these commands.
print("********************Part B - Task 2************************")

# 1: assert
# is sanity-check. An expression is tested, and if the result comes up false, an exception is raised.
# It is useful when you need validation of input, or check your output of function.
print("**** Part B - Task 2 : Assert ****")
number=int(input('Enter a negative number:')) # take negative integer input from user.
assert(number<0),'Only negative numbers are allowed!'  #If it s, positive raise an error!

#2: Break and Continue
#Just play with this friend:
print("**** Part B - Task 2 : Break and Continue ****")
while(1==1):   ##create an infinite loop to play with a friend.
    number=int(input('Guess a number to break the loop:'))
    if(number%5 == 0):
        print("Good Job!")
        break     ##when if condition is true; you break the loop chain.
    else:
        continue   ##if you dont know the rule of breaking, the loop continue forever.

print("**** End of Part B - Task 2  ****")
############################################################################################
# Task-3: Initialize a single dimensional array of size 21 with random integers from 10 to 99.
# Then please provide a new 2-D array (this time to two-dimensional one) of size 100 (row)
# and 6 (column) to composed of integers from 1 to 54 (just like Süper Loto).
# 3 5 22 34 39 41
# 2 12 23 45 46 53
# 1 3 6 13 18 32
# Now here are the challanges:
# The integers in each row should be in ascending order.
# There should be no repetition in a row. If there is you shall be changing the second by a new random integer.
print("********************Part B - Task 3************************")
import numpy as np
#initialize a single dimensional array:
a_21=np.random.randint(10,99,21,dtype="int")
print(a_21, "len of single dim array:",len(a_21))


#provide a new 2D array of size 100 row and 6 column, integer from 1 to 54:
print("**** Start to play with 2D Array ****")
a=np.random.randint(1,54,600,dtype="int").reshape(100,6)
print("Len of numpy array",len(a),"dim of array:",a.ndim,"shape of array:",a.shape,"size of array:",a.size)

# sort the rows:
a_sorted=np.sort(a)
#check first rows of orginal and sorted array if it is sorted:
print("orginal:", a[1])
print("orginal sorted:", a_sorted[1])

#there shuld be no repetition in a row.
#convert numpy array to pandas dataframe:
import pandas as pd
df=pd.DataFrame(a_sorted)
print(df,"**********************")

#clean duplicate values in rows of dataframe:
for i in range(0,100):  #key = column, i = row; travel in dataframe row by row.
    prev=-1
    for key in df.keys():
        if(key !=0):  # i am looking at previous values so -1 is not an index !
            prev=df.loc[i][key-1]  # take previous value in row
            if(prev == df.loc[i][key]): #if prev is equal to current value:
                #print("prev:",prev, "current:",df.loc[i][key])
                df.loc[i][key] = np.random.randint(1,54,1,dtype="int")    #change current value with new random integer
                #print("prev:",prev, "current:",df.loc[i][key])

#convert df into numpy array to sort again, because we assign random values to clean duplicate values:
a=np.array(df)
a_sorted=np.sort(a)
print("The last version of 2D array, sorted, no duplicate in rows:", a_sorted)

print("**** End of Part B - Task 3 ****")
print("**************End of Part B********************")
print("******************Beginning of Part C*******************")

# Task-1: Write a program to find the factorial of a given integer provided by the user. You do not have to do it by a function.
# Just ensure that you get input as an integer. If not please give warning to the user about the data type incompatibility.
# (Hint: Get the input from user and get the factorial by simple if, else and one loop).
# Please provide two functions to calculate the given integers for factorial by iteration and recursion.
# Then please compare their performance in time for the 5 integers (for instance: 10, 100, 1000, 10000, 100000).
# Then please briefly report your findings: which has higher performance, in other words, quicker?
print("***********Part C - Task 1**************")
import time
#factorial iteration:
def factorial_iter(n):
    m = 1
    while n>1:
        m=m*n
        n-=1
    return m

#factorial recursive:
def factorial_rec(n):
    if (n == 1):
        return 1
    else:
        return n*factorial_rec(n-1)

while(1==1):
    print("Enter -1 if you dont want to continue: ")
    if(x==-1):
        break
    x = int(input("Please enter an integer value for the increment:"))
    start_time = time.clock() #time.time()
    print(factorial_iter(x))
    print("factorial_iter:",time.clock() - start_time)
    start_time =time.clock() #time.time()
    print(factorial_rec(x))
    print("factorial_rec:",time.clock() - start_time)


# Then please briefly report your findings: which has higher performance, in other words, quicker?
print("Iterative method is faster than Recursive method and Recursive method has restriction on maximum recursion depth exceed so; we cannot use recursion method"
      "with big numbers")
print("***********End of Part C - Task 1**************")

# Task-2: Write a program to which makes computer to generate a random integer in an interval. User will try to guess it correctly.
# Program also should tell if the guess should be ‘lower’ or ‘higher’
# Tip: First of all check if the input is an integer. You can use number.isdigit() for this.
# Then compare the input to randomly generated number by using if,elif and else(to see if its lower,higher or equal).
print("***********Beginning of Part C - Task 2**************")
x=np.random.randint(1,100,1,dtype="int")
while(1==1):   ##create an infinite loop to play.
    number=int(input('Enter your guess:(Between 1 to 100)'))
    count=0
    if(number==x):
        print("Good Job!")
        print("you got it in %d tries!",count)
        break
    else:
        if(number>x):
            print("Lower")
            count+=1
        else:
            print("Higher")
            count+=1
print("***********End of Part C - Task 2**************")

#Write a program for the game ‘rock, paper and scissors’. It will be a two-player game, so get two inputs from two players.
#(Hint: Do several if, elifs to compare win situations of the inputs (there should be 4 in total, and 1 else to get a valid input))
#Result should be like in the following figure:
print("***********Beginning of Part C - Task 3**************")

first_p= input("First player, enter your choice: rock,paper or scissors?(lower letters pls)")
second_p= input("Second player, enter your choice: rock,paper or scissors?(lower letters pls)")

if(first_p==second_p):
    print("Tie!")
elif first_p == "rock":
        if second_p == "paper":
            print("Paper wins!")
elif first_p == "paper":
    if second_p == "scissors":
                    print("Scissors wins!")
elif first_p == "scissors":
    if second_p == "rock":
        print("Rock wins!")
else:
    print("That's not a valid input. Check your spelling!")
print("***********End of Part C - Task 3**************")
print("**************End of Part C********************")
print("******************Beginning of Part D*******************")

#Task-1: Please inspect through the following code in Python. Try to understand and explain the algorithm.
print("***********Part D - Task 1**************")
# Program to: COMMENT HERE PLEASE
lower = int(input("Enter lower range: ")) #Take an integer from user and assign it to lower variable.
upper = int(input("Enter upper range: ")) #Take an integer from user and assign it to upper variable.
# Print lower and upper variables taken from user with a format:
print("Numbers in the interval " + str(lower) + " - " + str(upper))

for num in range(lower, upper + 1):
    order = len(str(num)) # COMMENT HERE PLEASE
    sum = 0 # initialize sum variable
    temp = num # assign value of num variable to temp variable
    while temp > 0: # loop when value of temp is greater than zero
        digit = temp % 10 # assign mod 10 of temp to digit variable
        sum += digit ** order # add power(digit,order) to sum variable
        temp //= 10 # assign temp  to the division value when temp divided by 10
        if num == sum: # if num is equal to sum value
            print(num) # print num

print("***********Part D - Task 2**************")

#reorder the list -ascending-, from smallest to largest.
def algorithm_1 (alist):
    length = len(alist) # take length of the given list
    for i in range(0,length): # loop the range from zero to length of list. (travel on list)
        for j in range(i,length): # loop the range from i(outer loop iteration) to length of list. (travel on rest of list)
            if alist[j] < alist[i]: # if the next value is smaller than current value (alist[i])
                tmp = alist[i] # assign current value to temp variable
                alist[i] = alist[j] # swap next value and current value in the list.
                alist[j] = tmp # assign temp to next value.   (for last 3 line; sawp the value of list[j] and list[i])


def algorithm_2 (alist):
    length = len(alist) # take length of the given list
    for i in range(length-1, 0, -1): # travel on list from last to first element.
        pos = 0 # assign 0 to pos variable (hold first index)
        for j in range(1, i+1): # travel on rest of list from 1 to i+1 th element. (except first element, rest of the list)
            if alist[j] > alist[pos]: # if j th element bigger than first element.
                pos = j # assign j to pos (hold the index)
        tmp = alist[i] # hold the value of current i th element
        alist[i] = alist[pos] # swap the value of pos th (j th) and ith element.
        alist[pos] = tmp # new value of pos index is tmp (alist[i])   # (for last 3 line; sawp the value of list[j] and list[i])
    return alist # return ascending ordered list

#compare them:
print("First algorithm travel list from first element to last element twice, it travels list twice for each time and there is no return on it."
      "Second algorithm travel list from last element to first element,it travels list like hold the last element and travel on rest of the list, "
      "for next iteration hold "
      "the next element and compare this to rest of list element and go on until hold first element. "
      " it holds last element of list for each iteration and compare the rest of list to this value. "
      "If the element is bigger than holded element, swap their position to reorder the list."
      " So, second one is better because first one is traveling all list for each iteration.")

#Task-3: Please provide sufficient explanation for the given code below. How does it work? What does it do?
print("***********Part D - Task 3**************")

# this is tic-tac-toe game.
import random as rd
board = [1,2,3,4,5,6,7,8,9] # create 3*3 tic-tac-toe board
def show_board(): # this functions show tic-tac-toe board; it prints board list elements in format of tic-tac-toe board.
    print (str(board[0])+" | "+str(board[1])+" | "+str(board[2]))
    print ('----------')
    print (str(board[3])+" | "+str(board[4])+" | "+str(board[5]))
    print ('----------')
    print (str(board[6])+" | "+str(board[7])+" | "+str(board[8]))
#show_board()
def winConditions(char, s1, s2, s3): # this functions hold the win conditions.
    if board[s1] == char and board[s2] == char and board[s3] == char:
        return True
def wins(char): # includes all combinations to win; takes input as x or o, call winConditions function with x/o
                #  and position combinations. return True if winConditions returns True.
    if winConditions(char, 0, 1, 2):
        return True
    if winConditions(char, 3, 4, 5):
        return True
    if winConditions(char, 6, 7, 8):
        return True
    if winConditions(char, 2, 4, 6):
        return True
    if winConditions(char, 0, 3, 6):
        return True
    if winConditions(char, 1, 4, 7):
        return True
    if winConditions(char,2, 5, 8):
        return True
    if winConditions(char, 0, 4, 8):
        return True

print("Welcome to tic-tac-toe. You will be playing as X \n")

def control():
    count = 0
    while True: #infinite loop until all elements of board is full (if they are full, then finish the game.)
        tile = input("Select an available tile between (1-9): ")
        move_1 = int(tile) # hold the position taken from user to move x.
        if board[move_1 - 1] != 'x' and board[move_1 - 1] != 'o':
            board[move_1 - 1] = 'x'
            count = count + 1
            #print(count)
            if wins('x'):
                print("X Wins!")
                break
            elif(count == 9):
                print("Tie")
                break
            while True: # infinite loop until board is full or o wins. Take random position for o, place it and show the board to user for his next move.
                move_2 = rd.randint(1,9)
                if board[move_2 - 1] != 'x' and board[move_2 - 1] != 'o':
                    board[move_2 - 1] = 'o'
                    count = count + 1
                    #print(count)
                    show_board()
                    if wins('o'):
                        print("O Wins!")
                    break
                break
        else:
            print("Spot Taken... Try again.")

control()
