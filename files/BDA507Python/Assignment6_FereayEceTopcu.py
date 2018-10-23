#1. Please construct a two-dimensional numpy array composed of random integers (in a range of -10 to
#38) with a size of 5 columns (cities A, B, C, D, and E) and 364 rows (days).

print("************** Ques 1******************")
import numpy as np
import pandas as pd

#cities[4,364]=0

#create random 2D Array
cities= np.random.randint(-10,39,size=(365,5))
print(cities)

#print("ndim:", cities.ndim)
#print("shape:", cities.shape)
#print("size: ", cities.size)

#Convert 2D array into DF
citiesdf=pd.DataFrame(cities,columns=['A','B','C','D','E'])
#print(citiesdf.head())
print(citiesdf.describe())
print(citiesdf.info())

#2. These are the measured temperatures and determine which of these 5 cities is the hottest one and
#which of these is the coldest one simply by your code. You should not decide these manually, in other
#words, your code should decide.
print("************** Ques 2******************")

avg_max=0
max_city='x'
avg_min=38
min_city='y'
for key in citiesdf.keys():
    if citiesdf[key].mean() >= avg_max:
        avg_max=citiesdf[key].mean()
        max_city=key
    if citiesdf[key].mean()<= avg_min:
        avg_min=citiesdf[key].mean()
        min_city=key

print("The hottest city is ",max_city, " : ",avg_max)
print("The coldest city is ",min_city," : ",avg_min)

#3. Figure out which is the hottest day (with a comparison through 5 cities) again by your Python
#code, please. Is it possible to look for a (statistically) significant difference? If so, how?

#http://blog.minitab.com/blog/understanding-statistics/what-can-you-say-when-your-p-value-is-greater-than-005
#http://hamelg.blogspot.com.tr/2015/11/python-for-data-analysis-part-16_23.html
#Notes:  If the p-value is less than 0.05, we reject the null hypothesis that there's no difference between the means
#and conclude that a significant difference does exist. If the p-value is larger than 0.05,
#we cannot conclude that a significant difference exists.
#That's pretty straightforward, right?  Below 0.05, significant. Over 0.05, not significant.

print("************** Ques 3******************")
from scipy import stats
max=0
#max_day_index=[]
max_index_list=pd.DataFrame()

for key in citiesdf.keys():
    if citiesdf[key].max() >= max:
        max=citiesdf[key].max()
        max_city=key
        max_day_index=np.where(citiesdf[key] == max)  ##there is more than one day with same max degree
        max_index_list.append(citiesdf.loc[max_day_index][key])
        #print("max_city: ",max_city,"max value: ",max,"max_day_index: ", max_day_index)
        #print("max days", citiesdf.loc[max_day_index][key])

print("max_city: ",max_city,"\nmax value: ",max,"\nyear of the days which has hottest degree: ", max_day_index)
#print("max days matrix", max_index_list)

##statistical approach with  ANOVA Test:
print(stats.f_oneway(citiesdf["A"],citiesdf["B"],citiesdf["C"],citiesdf["D"],citiesdf["E"]))
#F_onewayResult(statistic=2.1229770435359732, pvalue=0.075593263960044843)
#print(stats.kruskal(citiesdf["A"],citiesdf["B"],citiesdf["C"],citiesdf["D"],citiesdf["E"]))
print('pvalue = 0.076  and The results were not statistically significant.')


#4. Please find a way (among one of the python libraries) to visualize/demonstrate these 364 days with
#a figure as an average of 5 cities. Hint: There should be 364 x 5 data points.

#http://pandas.pydata.org/pandas-docs/version/0.15.0/visualization.html#scatter-plot

print("************** Ques 4******************")
import matplotlib.pyplot as plt

#365 * 5 points:

x=np.arange(0,365)
plt.subplot(221)
plt.scatter(x, citiesdf.loc[:,"A"], c="g", alpha=0.5, marker=r'$\clubsuit$',
            label="City A")
plt.scatter(x, citiesdf.loc[:,"B"], c="r", alpha=0.5, marker="d",
            label="City B")
plt.scatter(x, citiesdf.loc[:,"C"], c="y", alpha=0.5, marker="P",
            label="City C")
plt.scatter(x, citiesdf.loc[:,"D"], c="b", alpha=0.5, marker="8",
            label="City D")
plt.scatter(x, citiesdf.loc[:,"E"], c="DarkRed", alpha=0.5, marker="2",
            label="City E")
plt.title("Daily Weather Report")
plt.xlabel("Days of Year")
plt.ylabel("Degree")
#plt.legend(loc=6)
plt.legend(bbox_to_anchor=(1, 1.05))  ##move legend to out of plot
plt.show()
plt.subplot(222)
plt.scatter(x, citiesdf.loc[:,"A"], c="g", alpha=0.5, marker=r'$\clubsuit$',
            label="City A")
plt.scatter(x, citiesdf.loc[:,"B"], c="r", alpha=0.5, marker="d",
            label="City B")
plt.scatter(x, citiesdf.loc[:,"C"], c="y", alpha=0.5, marker="P",
            label="City C")
plt.scatter(x, citiesdf.loc[:,"D"], c="b", alpha=0.5, marker="8",
            label="City D")
plt.scatter(x, citiesdf.loc[:,"E"], c="DarkRed", alpha=0.5, marker="2",
            label="City E")
plt.title("Daily Weather Report")
plt.xlabel("Days of Year")
plt.ylabel("Degree")
#plt.legend(loc=6)
plt.legend(bbox_to_anchor=(1, 1.05))  ##move legend to out of plot
plt.show()





# x=np.arange(0,365)
# #print(x)
# fig = plt.figure()
# ax1 = fig.add_subplot(111)
# ax1.scatter(x,citiesdf.loc[:,"A"],c='b',marker="s",label='A')
# ax1.scatter(x,citiesdf.loc[:,"B"],c='c',marker="o",label='B')
# ax1.scatter(x,citiesdf.loc[:,"C"],c='y',label='C')
# ax1.scatter(x,citiesdf.loc[:,"D"],c='b',label='D')
# ax1.scatter(x,citiesdf.loc[:,"E"],c='r',marker="*",label='E')
# plt.show()

##5. Suppose that these 364 days are 52 weeks. Then please provide a weekly (average) illustration of
#these cities and illustrate them via figures/plots. Hint: there should be 52 x 5 data points.
#https://pandas.pydata.org/pandas-docs/stable/dsintro.html#series-is-ndarray-like
#https://matplotlib.org/devdocs/gallery/lines_bars_and_markers/scatter_symbol.html#sphx-glr-gallery-lines-bars-and-markers-scatter-symbol-py
#https://matplotlib.org/api/markers_api.html
#print(citiesdf.loc[range(0,52),"A"].mean())

print("************** Ques 5******************")

#create an empty dataframe.
df=pd.DataFrame( index=range(1,53), columns=['A', 'B','C','D','E'])

#fill the dataframe with means of cities for each week.
for key in citiesdf.keys():
    for i in range(1,53):
        df.loc[i,key]=citiesdf.loc[range(i,i*7),key].mean()
        #print(key, "quarter",i,citiesdf.loc[range(i,i*52),key].mean())  #for(1,5)
        #print(key, "week",i,citiesdf.loc[range(i,i*7),key].mean())
#print(df)

#graph it!
x=np.arange(1,53)
plt.subplot(224)
plt.scatter(x, df.loc[:,"A"], c="g", alpha=0.5, marker=r'$\clubsuit$',
            label="City A")
plt.scatter(x, df.loc[:,"B"], c="r", alpha=0.5, marker="d",
            label="City B")
plt.scatter(x, df.loc[:,"C"], c="y", alpha=0.5, marker="P",
            label="City C")
plt.scatter(x, df.loc[:,"D"], c="b", alpha=0.5, marker="8",
            label="City D")
plt.scatter(x, df.loc[:,"E"], c="DarkRed", alpha=0.5, marker="2",
            label="City E")
plt.title("Weekly Average Weather Report")
plt.xlabel("Weeks of Year")
plt.ylabel("Degree")
plt.legend(loc=4)
plt.show()

#print(df.loc[1,"E"])


#x=np.arange(1,53)
#fig = plt.figure()
#ax1 = fig.add_subplot(111)
#ax1.scatter(x,df.loc[:,"A"],c='b',marker="s",label='A')
#ax1.scatter(x,df.loc[:,"B"],c='DarkBlue',label='B')
#ax1.scatter(x,df.loc[:,"C"],c='c',marker="o",label='C')
#ax1.scatter(x,df.loc[:,"D"],c='y',marker="*",label='D')
#ax1.scatter(x,df.loc[:,"E"],c='r',marker="s",label='E')
#plt.show()


##birden fazla grafiği ayrı ayrı çizmek için:
# import matplotlib.pyplot as plt
# f1 = plt.figure()
# f2 = plt.figure()
# ax1 = f1.add_subplot(111)
# ax1.plot(range(0,10))
# ax2 = f2.add_subplot(111)
# ax2.plot(range(10,20))
# plt.show()


