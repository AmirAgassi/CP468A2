import matplotlib.pyplot as plt
import csv
import math
import random 

random.seed(42)

def most_frequent(List):
    return max(set(List), key=List.count)

# Define the knn function first
def knn(data,new,k,percent):
    
    for i in range(len(data[0])):  # Adjusted for proper index access
        plt.scatter(data[0][i], data[1][i], c=data[2][i], s=21)
     
    while len(new[0])!=0: 
        x1=new[0][0]
        y1=new[1][0]
        distance=[]
        for j in range((len(data[0]))):
            x2=data[0][j]
            y2=data[1][j]

            sum=math.sqrt(pow((x1-x2),2) + pow((y1-y2),2))
            distance.append(float(sum))
        count=0
        max=0
        index=0
        min_indices = sorted(enumerate(distance), key=lambda x: x[1])[:k]
        min_positions = [x[0] for x in min_indices]  # Their indices
 
        color_index=[]
        for z in min_positions:
    
            color_index.append(data[2][z])
        color=most_frequent(color_index)    

        
        blue = plt.scatter([], [], c='blue', s=21, label='Not interested')  # Dummy scatter for blue dots
        red = plt.scatter([], [], c='red', s=21, label='Interested')  # Dummy scatter for red dots
        
        plt.scatter(x1,y1,c=color,edgecolors='black')
            
        plt.title(f"Customer Behavior: k-NN (k={k}) (split={percent}%)")
        plt.xlabel("Average amount spent per Purchases (in $)")
        plt.ylabel("Frequency of purchases per month")
        plt.legend(handles=[blue,red], loc="lower right")
        plt.ylim(0,3)
        
        # removing data from test set to train set
        numx= new[0].pop(0)
        numy= new[1].pop(0)
        data[0].append(float(numx))
        data[1].append(float(numy))
        data[2].append(color)
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

# spliting the csv file into 3 arrays whoch are the x coordiante y coordiants and color
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
        
# K Nearest Neighbour k=1 

#80% (16 train, 4 test)
x1_train = x1[:16]
x2_train = x2[:16]
color_train = color[:16]


x1_test = x1[-4:]
x2_test = x2[-4:]
color_test = color[-4:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,1,80)

#60% (12 train,8 test)
x1_train = x1[:12]
x2_train = x2[:12]
color_train = color[:12]


x1_test = x1[-8:]
x2_test = x2[-8:]
color_test = color[-8:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,1,60)

#50% (10 train, 10 test)

x1_train = x1[:10]
x2_train = x2[:10]
color_train = color[:10]


x1_test = x1[-10:]
x2_test = x2[-10:]
color_test = color[-10:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,1,50)


# K Nearest Neighbour k=2

#80% (16 train, 4 test)
x1_train = x1[:16]
x2_train = x2[:16]
color_train = color[:16]


x1_test = x1[-4:]
x2_test = x2[-4:]
color_test = color[-4:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,2,80)

#60% (12 train,8 test)
x1_train = x1[:12]
x2_train = x2[:12]
color_train = color[:12]


x1_test = x1[-8:]
x2_test = x2[-8:]
color_test = color[-8:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,2,60)

#50% (10 train, 10 test)

x1_train = x1[:10]
x2_train = x2[:10]
color_train = color[:10]


x1_test = x1[-10:]
x2_test = x2[-10:]
color_test = color[-10:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,2,50)

# K Nearest Neighbour k=3

#80% (16 train, 4 test)
x1_train = x1[:16]
x2_train = x2[:16]
color_train = color[:16]


x1_test = x1[-4:]
x2_test = x2[-4:]
color_test = color[-4:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,3,80)

#60% (12 train,8 test)
x1_train = x1[:12]
x2_train = x2[:12]
color_train = color[:12]


x1_test = x1[-8:]
x2_test = x2[-8:]
color_test = color[-8:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,3,60)

#50% (10 train, 10 test)

x1_train = x1[:10]
x2_train = x2[:10]
color_train = color[:10]


x1_test = x1[-10:]
x2_test = x2[-10:]
color_test = color[-10:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,3,50)

# K Nearest Neighbour k=4

#80% (16 train, 4 test)
x1_train = x1[:16]
x2_train = x2[:16]
color_train = color[:16]


x1_test = x1[-4:]
x2_test = x2[-4:]
color_test = color[-4:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,4,80)

#60% (12 train,8 test)
x1_train = x1[:12]
x2_train = x2[:12]
color_train = color[:12]


x1_test = x1[-8:]
x2_test = x2[-8:]
color_test = color[-8:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,4,60)

#50% (10 train, 10 test)

x1_train = x1[:10]
x2_train = x2[:10]
color_train = color[:10]


x1_test = x1[-10:]
x2_test = x2[-10:]
color_test = color[-10:]

# Create training and test datasets
dataset_train = [x1_train, x2_train, color_train]  # Training set (16 points)
dataset_test = [x1_test, x2_test, color_test]  # Test set (4 points)

knn(dataset_train, dataset_test,4,50)
