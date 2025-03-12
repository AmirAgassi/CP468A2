import matplotlib.pyplot as plt
import csv
import math
# Define the knn function first
def knn(data, new):
    x1=new[0]
    y1=new[1]
    distance=[]
    for i in range((len(data[0]))):
        x2=data[0][i]
        y2=data[1][i]

        sum=math.sqrt(pow((x1-x2),2) + pow((y1-y2),2))
        distance.append(float(sum))
    count=0
    max=0
    index=0
    distance_index = distance.index(min(distance))
   
    
    for i in range(len(data[0])):  # Adjusted for proper index access
        plt.scatter(data[0][i], data[1][i], c=data[2][i], s=21)
    
    blue = plt.scatter([], [], c='blue', s=21, label='Not interested')  # Dummy scatter for blue dots
    red = plt.scatter([], [], c='red', s=21, label='Interested')  # Dummy scatter for red dots
    plt.scatter(x1,y1,c=data[2][distance_index])
    plt.title("Customer Behavior: Premium Membership Interest vs. Spending and Frequency KNN =1")
    plt.xlabel("Average amount spent per Purchases (in $)")
    plt.ylabel("Frequency of purchases per month")
    plt.legend(handles=[blue,red], loc="lower right")
    plt.ylim(0,3)
    plt.show()

# Read the CSV file and process the data
filename = "CustomerDataset_Q1.csv"

fields = []
rows = []

with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    
    for row in csvreader:
        rows.append(row)

dataset = []

for row in rows:
    hold = []
    for col in row:
        hold.append(col)
    dataset.append(hold)

x1 = []
x2 = []
y = []

for data in dataset:
    x1.append(float(data[0]))  # Make sure to convert to float
    x2.append(float(data[1]))  # Make sure to convert to float
    y.append(data[2])

color = []
for value in y:
    if value == '0':
        color.append('blue')
    else:
        color.append('red')

x1plot = x1[0:19]
x2plot = x2[0:19]
colorplot = color[0:19]

dataset=[x1plot,x2plot,colorplot]
new_data = [x1[-1], x2[-1], color[-1]]


for i in range(len(x1)):
    plt.scatter(x1[i],x2[i], c=color[i], s=21)
blue = plt.scatter([], [], c='blue', s=21, label='Not interested')  # Dummy scatter for blue dots
red = plt.scatter([], [], c='red', s=21, label='Interested')  # Dummy scatter for red dots
plt.title("Customer Behavior: Premium Membership Interest vs. Spending and Frequency")
plt.xlabel("Average amount spent per Purchases (in $)")
plt.ylabel("Frequency of purchases per month")
plt.legend(handles=[blue,red], loc="lower right")
plt.ylim(0,3)
plt.show()


# Call the knn function with dataset and new_data
knn(dataset, new_data)
