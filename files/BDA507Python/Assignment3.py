print("**************************** Task 1 *******************************")
##Please write a function to find the minimum number among three given integers. Your function
##Then you should call the function with different integer sets in order to check whether it worksproperly.

def min_of_two(x,y):
    if x > y:
        return(y)
    else:
        return(x)

def min_of_three(x,y,z):
    return(min_of_two(min_of_two(x,y),z))

print(min_of_three(2,3,4))
print(min_of_three(7,6,5))
print(min_of_three(1,9,1))
print(min_of_three(1,9,2))
print(min_of_three(-1,-19,2))

print("**************************** Task 2 *******************************")
# 1)Please get 6 input numbers (as float or as double) from the user as the coordinates of the three
# points. These points correspond to a triangle. Please calculate the length of each side of this
# triangle (you can use the distance formula for the points). 2)You should provide a distance
# formula for this calculation. 3)Then you should provide an area function and calculate the area
# of this triangle. (Hint: You can use the U-formula for this). You should then decide whether this
# is an obtuse, right or acute triangle (as in the previous assignment).
import math
def find_distance(a,b,c,d):
    distance = math.sqrt(((a-c)**2)+((b-d)**2))
    return(distance)

def area_of_triangle(a,b,c):
    u=(a+b+c)/2
    t_area= math.sqrt(u*(u-a)*(u-b)*(u-c))
    return t_area

def check_triangle(a,b,c):
    max_side = max(a,b,c)

    if (max_side== c and c>=a+b):
        print("It cannot be a triangle! 1")
    elif(max_side==a and a>=b+c):
         print("It cannot be a triangle! 3")
    elif(max_side==b and b>=a+c):
         print("It cannot be a triangle! 5 ")
    else:
         if(a*a + b*b == c*c or a*a+c*c==b*b or b*b+c*c==a*a ):
             return("it s a right-dik- angle!")
         elif(a*a + b*b < c*c or a*a+c*c<b*b or b*b+c*c<a*a):
             return("it s an acute -daracili- triangle!")
         else:
             return("it s an obtuse-genisacili- triangle!")

#1
x1 = float(input("x1: "))
y1 = float(input("y1: "))
x2 = float(input("x2: "))
y2 = float(input("y2: "))
x3 = float(input("x3: "))
y3 = float(input("y3: "))
#2
side1 = find_distance(x1,y1,x2,y2)
side2 = find_distance(x2,y2,x3,y3)
side3 = find_distance(x1,y1,x3,y3)
print("Side1:",side1,"Side 2: ",side2, "Side 3: ",side3)
print("Area of this triangle: ", area_of_triangle(side1,side2,side3))
print(check_triangle(side1,side2,side3))

print("**************************** Task 3 *******************************")
#You have such a list of strings:color = ['One', 'Two', 'Three', 'Four', 'Five', 'Six']Your function or script should put this list
#inalphabetical order than write this ordered list intoa new text file. Then it should open the list and print them on the console.
import os.path

def order_list (slist):
    slist.sort()
    return(slist)

color = ['One', 'Two', 'Three', 'Four', 'Five', 'Six']

if os.path.isfile('List.txt') == False:
    open('List.txt',"w+")

with open('List.txt', "w") as w_file:
        for c in order_list(color):
                w_file.write("%s\t" % c)

content = open('List.txt')
print(content.read())

print("**************************** Task 4 *******************************")
# Your code should look for the amino acid patterns like
# (ACCCA or ACCA or ABA) and (AAABB or BBBBBAAACCC).in a huge random string
# of A, B, and C’s. You can set up your own amino acid sequence for testing.
# You are suggested to use the library for regular expressions.
# Here is a sample sequence to test:AAACACCCCBAAAAACBBBBABBCCCABAAABBAAAAAAAAAABBBBBAAABCCCACCCCBBBBBAAA is sufficient
# to have a return value as true or false for the time being.

import re
def text_match(text):
        patterns = ['((AC{2,3}A$)|(AB?A))','(A-B){1,}C{0,3}$']
            # ilk part bu : '(AC{2,3}A$)|(AB?A)'   || '((AC{2,3}A$)|(AB?A))' -- bu daha doğru ilk pattern için.
        i=0
        while i<len(patterns):
            if re.search(patterns[i],  text):
                    return True
            else:
                    return False
            i+=1

print(text_match('AAACACCCCBAAAAACBBBBABBCCCABAAABBAAAAAAAAAABBBBBAAABCCCACCCCBBBBBAAA'))
print(text_match("ACCCA"))
print(text_match("ACCA"))
print(text_match("ABA"))
print(text_match("ACA"))
print(text_match("AxbbbbbbbbbA"))
print(text_match("AAABB"))
print(text_match("BBBBBAAACCC"))

print("**************************** Task 5 *******************************")
#You should make the file named “BMI_data.txt” read in python 3.
#Then you are required to calculate the BMI values for each of the participant as a new column
#at the rightmost part as “ID Name Surname Gender Height Weight BMI”.
#Calculate the BMI value for each participant and write these into a new array named
#BMI_value (You do not need to have the header but just the calculated values in it). Finally
#provide a figure that include these people (showing their BMI values only).
#1 Read file


import matplotlib.pyplot as plt
data=[]
f=open('BMI_data.txt',"r+")
header_list = f.readline().split() #take headers into a list.
lines=f.readlines()
for i in lines:
    i = i.strip()  #clean the \n.
    col = i.split()  # split the line into words.
    person_info_dict={}
    j=0
    while j<len(header_list):   ##convert line into dict. (Like an hashtable. Key-Value pairs.)
        person_info_dict[header_list[j]]=col[j]
        person_info_dict["BMI"]=0.0  ##create a new column as BMI
        j+=1
    data.append(person_info_dict) ##append dict to list; so for each row we have header-data pairs.
    #print(person_info_dict)
#print(data)
#2 Calculate BMI and write in data list as BMI for each person.
# BMI = weight (kg) / [height (m)]2
j=0
while j<len(data):
    #print(float(data[j]["BMI"]))
    #print(float(data[j]["Height"]))
    data[j]["BMI"] = round((float(data[j]["Weight"])/float(data[j]["Height"]))**2,2)
    j+=1
#print(data)

#print new list with BMI_values for each person.
BMI_value = data
x=[]
y=[]
for i in BMI_value:
   x.append(i["BMI"])
   y.append(i["ID"])
   #print(i)
print(x)
print(y)
plt.plot(y,x)
plt.xlabel('x - axis')
# Set the y axis label of the current axis.
plt.ylabel('y - axis')
# Set a title
plt.title('Sample graph!')
# Display a figure.
plt.show()






