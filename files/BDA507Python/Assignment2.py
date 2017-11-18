print("\n**************Task 1*****************\n")
#1. Please write a function in python 3 to calculate the absolute value of a given number (without
#using the abs() function of course). The absolute value is important for distance, as you might
#probably remember. You should provide a proper name for the function and one argument. It
#should print the absolute value of the given integer on the console.
def abs(x):
    abv=0
    if x >= 0:
         abv=x
    else:
         abv=-x
    print("Absolute value of",x,"is",abv)
abs(5)
abs(-15)

print("\n**************Task 2*****************\n")
#2. Please write a function in python 3 to check whether given 3 numbers as the sides of a triangle
#are eligible to provide a triangle. We all know that the side lengths such as 3, 11, 15 cannot
#make a triangle, since 15 is greater than the sum of other numbers. In addition to this, you are
#also check if this triangle is a right, acute, or obtuse triangle.

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
             print("it s a right-dik- angle!")
         elif(a*a + b*b < c*c or a*a+c*c<b*b or b*b+c*c<a*a):
             print("it s an acute -daracili- triangle!")
         else:
             print("it s an obtuse-genisacili- triangle!")

check_triangle(3,11,15)
check_triangle(3,2,1)
check_triangle(3,8,1)
check_triangle(3,4,5)
check_triangle(5,15,15)
check_triangle(5,10,13)


print("\n**************Task 3*****************\n")
#3. There are three Boolean objects. As you should remember, Boolean objects hold True or False
#values. You would like to calculate how many of these three objects hold True values. You are
#advised to use if-elif-else conditions for these calculations. You do not have to provide a
#function for this but you are recommended to do so. Let’s assume that x, y, z = True, False,
#False. Your function should give the output 1 for the number of True objects and will write x
#as the True objects and others as False.
def check_boolean(x,y,z):
    cnt=0;
    if __name__ == '__main__':
        if(x and y and z):
            print('All of variables are true: X Y Z')
            cnt+=3
        elif(x and y):
            print('Just two variables are true:X and Y; and Z is false.')
            cnt+=2
        elif(x and z):
            print('Just two variables are true:X and Z; and Y is false.')
            cnt+=2
        elif(y and z):
             print('Just two variables are true:Y and Z; and X is false.')
             cnt+=2
        elif(x):
            print('Just X is True; Y and Z are False.')
            cnt+=1
        elif(y):
            print('Just Y is True; X and Z are False.')
            cnt+=1
        elif(z):
            print('Just Z is True; X and Y are False.')
            cnt+=1
        else:
            print('All of variables are False: X Y Z')
    print("Number of True objects:",cnt)

x, y, z = True, False,False
check_boolean(x,y,z)
x, y, z = False, True,False
check_boolean(x,y,z)
x, y, z = False,False,True
check_boolean(x,y,z)
x, y, z = True, True,False
check_boolean(x,y,z)
x, y, z = False, True,True
check_boolean(x,y,z)
x, y, z = True, True,True
check_boolean(x,y,z)
x, y, z = False, False,False
check_boolean(x,y,z)

print("\n**************Task 4*****************\n")
#4.You are given a list of names as in the A1. We can directly bring it as follows:
#nameList = ["Utku", "Aynur", "Tarik", "Aktan", "Asli", "Ahmet", "Metin", "Funda", "Kemal",
#"Hayko", "Zelal", "Kenan", "Asli", "Atakan", "Umut"]
#You are asked to check if there is “Asli” in this list and please calculate how many of these. (If
#the name is in the list please print it as “yes in the list” else “not in the list”).
#Then check for “Kemalettin” in this list and please indicate how many of them there are in the list.
#To calculate the total number of names. Hint: There are 15 names in this list but how to calculate it with a for/while loop.
#To calculate the number of names that contain the letter a (uppercase a or lowercase a do not matter).

nameList = ["Utku", "Aynur", "Tarik", "Aktan", "Asli", "Ahmet", "Metin", "Funda", "Kemal","Hayko", "Zelal", "Kenan", "Asli", "Atakan", "Umut"]

name_cnt=0
asli_cnt=0
kml_cnt=0
a_cnt=0
for i in nameList:
    name_cnt+=1

    if(i == "Asli"):
        asli_cnt+=1
    if(i=="Kemalettin"):
        kml_cnt+=1
    if(i.__contains__("a") or i.__contains__("A")):
        a_cnt+=1
        #print("Found an a! ", i)
if(asli_cnt>0):
    print("Yes Asli in the list and the occurance of Asli is",asli_cnt)
else:
    print("Asli is not in the list.")

if(kml_cnt>0):
    print("The occurance of Kemalettin is",kml_cnt)
else:
    print("Kemalettin is not in the list.")
print("the total number of names:",name_cnt)
print("the number of names that contain the letter a:",a_cnt)

print("\n**************Task 5*****************\n")
#5. You are asked to calculate the list and sum of numbers divisible by 13 and that are odd (evens
#are excluded) from 100 to 999 by using a for/while loop. For instance: 105 is both divisible by
#13 and is an odd number but not 118 (which is divisible by 13 but an even number). You may
#create an empty list object, append each tested item and sum the content at the very final step
nl = []
for x in range(100, 999):
    if (x % 13 == 0) and (x % 2 == 1):
        nl.append(x)
sum=0
for i in nl:
    #print(i)
    sum+=i
print("The sum of numbers:",sum)

print("\n***************Bonus******************\n")
#Bonus Perfect Number:
def check_perfect_number(n):
    sum = 0
    for i in range(1, n):
        if n % i == 0:
            sum += i
    if(sum==n):
       print(n,"is a perfect number!")
    else:
       print(n,"is not a perfect number!")
check_perfect_number(28)
check_perfect_number(24)



