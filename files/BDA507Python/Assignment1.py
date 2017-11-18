# Task 1: calculate area of triangle:
print("\n**************Task 1*****************\n")
height = 10
base = 5
area = 1
area = (height * area)/ 2
print("the area of triangle: ", area)


#Task 2: Gen (the count of A, C, T, and G)
print("\n**************Task 2*****************\n")
geneSequence ="AATCGGCATGCCGAATTTCCGCTATGTTGCATGCATCGTACGATGCATATGCATAGAGGxGCTTTTAACGATGCCCGATGATTTCATGCCCGTAACGACTCTGACGTACTG"
listGen=list(geneSequence) # convert str(geneSequence) into list
countA =listGen.count("A")
countC=listGen.count("C")
countT=listGen.count("T")
countG=listGen.count("G")
lenGen=len(listGen)
print("total of nucleotides: ",countA+countC+countT+countG) #find total of nucleotides
print("length of list: ",lenGen)  #find length of geneSeq
listGen.pop(listGen.index("x"))  #find x and remove it from list.
lenGen=len(listGen)
#print(listGen)
print("new length after popped x: ",lenGen)

#Task 3:
print("\n**************Task 3*****************\n")
nameList = ["Utku", "Aynur", "Tarik", "Aktan", "Asli", "Ahmet", "Metin", "Funda", "Kemal","Hayko", "Zelal", "Kenan", "Asli", "Atakan", "Umut"]
print("Index of Aktan: ",nameList.index("Aktan"))
#print("Index of Ezgi: ",nameList.index("Ezgi"))   #i dont want to see an error so i comment it !!!
print("Total number of students: ",len(nameList))
nameList.append("Yahya")
nameList.remove("Ahmet")
nameList.sort()
print("Sorted list: ",nameList)
print("Last element of list: ",nameList[len(nameList)-1])


#Task 4: Compare the lists
print("\n**************Task 4*****************\n")
Listx1 = ["w", 3, "r", "e", "s", "o", "u", "r", "c", "e"]
Listx2 = ["w", 3, "e", "r", "e", "s", "o", "u", "r", "c"]
Listx1.pop(Listx1.index(3))   # 3 is integer and cannot sort string and integer together so pop it from lists.
Listx2.pop(Listx2.index(3))
print("Are the lists equal?",(Listx1)==(Listx2))  #Lists are not equal so we should sort them now.
Listx1=sorted(Listx1)   # Sort the lists without 3.
Listx2=sorted(Listx2)
Listx1.append(3)  ##append 3 on end of the lists. So they are same with orginal lists but sorted.
Listx2.append(3)
print("Are the lists equal?",(Listx1)==(Listx2))   # now, they are equal.

#Task 5:
print("\n**************Task 5*****************\n")
numberList = [5, 10, 3, 25, 7, 4, 15, 13, 8, 4, 6]
print("The highest value: ",max(numberList))
print("The lowest value: ", min(numberList))
print("The range: ", max(numberList), "-",min(numberList) )
print("Count of numbers: ",len(numberList))
print("Sum of elements: ",sum(numberList))
print("Avg of elements: ",sum(numberList)/len(numberList))
