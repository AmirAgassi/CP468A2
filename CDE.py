import matplotlib.pyplot as plt
import csv
import math
import random

random.seed(42)

def most_frequent(List):
    return max(set(List), key=List.count)

def knn(data, new, k, train_percent):
    correct = 0  # Track correct predictions
    total = len(new[0])
    
    for i in range(len(data[0])):
        plt.scatter(data[0][i], data[1][i], c=data[2][i], s=21)
    
    predictions = []
    actuals = new[2].copy()
    
    while len(new[0]) != 0:
        x1, y1 = new[0][0], new[1][0]
        distances = [math.sqrt((x1 - data[0][j])**2 + (y1 - data[1][j])**2) for j in range(len(data[0]))]
        
        min_indices = sorted(enumerate(distances), key=lambda x: x[1])[:k]
        min_positions = [x[0] for x in min_indices]
        color_index = [data[2][z] for z in min_positions]
        color = most_frequent(color_index)
        
        predictions.append(color)
        if color == actuals.pop(0):
            correct += 1
        
        plt.scatter(x1, y1, c=color, edgecolors='black')
        new[0].pop(0)
        new[1].pop(0)
        data[0].append(x1)
        data[1].append(y1)
        data[2].append(color)
    
    plt.title(f"k={k}, Train={train_percent}%")
    plt.show()
    return correct / total  # Compute accuracy

def evaluate_knn(x1, x2, color, k_values, splits):
    results = []
    
    for k in k_values:
        for percent in splits:
            train_size = int(len(x1) * (percent / 100))
            x1_train, x2_train, color_train = x1[:train_size], x2[:train_size], color[:train_size]
            x1_test, x2_test, color_test = x1[train_size:], x2[train_size:], color[train_size:]
            
            dataset_train = [x1_train, x2_train, color_train]
            dataset_test = [x1_test, x2_test, color_test]
            
            accuracy = knn(dataset_train, dataset_test, k, percent)
            results.append((k, percent, accuracy))
            
    
    return results

filename = "CustomerDataset_Q1.csv"
fields, rows = [], []
with open(filename, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    fields = next(csvreader)
    for row in csvreader:
        rows.append(row)

dataset = [[float(row[0]), float(row[1]), 'blue' if row[2] == '0' else 'red'] for row in rows]
x1, x2, color = zip(*dataset)

k_values = [1, 2, 3, 4]
splits = [80, 60, 50]
results = evaluate_knn(list(x1), list(x2), list(color), k_values, splits)

# Display Results Table
print("\nFinal Accuracy Results:")
print("k | Train % | Accuracy")
print("----------------------")
for k, percent, acc in results:
    print(f"{k} |  {percent}%    | {acc:.2f}")
