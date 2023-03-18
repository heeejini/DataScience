from matplotlib import pyplot as plt
import  numpy  as np

#ex-1
wt=[]
ht=[]
bmi =[]
ht = np.random.randint(140,200,size=100)*0.01 # centimeter convert to meter
wt=np.random.uniform(40.0,90.0,size=100)

wt=np.array(wt)
ht=np.array(ht)

#print(wt)
#print(ht)

bmi = np.array(wt / (ht**2))
print("\nBMI for the 100 studnets : ")
print(bmi)
print("\nFirst 10 elements of the bmi array:")
print(bmi[:10])

#ex2

#draw the bar chart, histogram, pie chart, and scatter plot

Underweight=Healthy=Overweight =Obese=0
langs = ['Underweight','Healthy','Overweight','Obese']

for i in bmi:
    if i<18.5:
        Underweight+=1
    elif 18.5<=i<25:
        Healthy+=1
    elif 25.0<=i<30:
        Overweight+=1
    elif 30.0<=i :
        Obese+=1
student = [Underweight, Healthy, Overweight,Obese]
#bar chart
plt.title("bar chart")
plt.bar(langs,student)
plt.show()

#histogram, bins=4
plt.hist(bmi, bins=4)
plt.title("historgram of result")
plt.xticks([18.5,25,30,40])
plt.xlabel('BMI')
plt.ylabel('number of students')
plt.show()

#pie chart
plt.title("pie chart")
plt.pie(student,labels=langs, autopct="%1.2f%%")
plt.show()

#scatter plot
plt.scatter(ht*100, wt)
plt.xlabel('Height')
plt.ylabel('Weight')
plt.title('Scatter Plot')
plt.show()



