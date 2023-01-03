import math
import csv

class Node:
    def __init__(self, attribute):
        self.attribute = attribute 
        self.children = [] 
        self.answer = "" 


def load_csv(filename):
    lines = csv.reader(open(filename, "r")) 
    dataset = list(lines)
    print(dataset)
    headers = dataset.pop(0) 
    return dataset, headers  
        
def subtables(data, col, delete):
    #print(data)
    dic = {}
    coldata = [ row[col] for row in data]
    attr = list(set(coldata)) 
    for k in attr:
        dic[k] = []
        #print(dic)
    for y in range(len(data)): 
        key = data[y][col]
        #print(key)
        
        if delete:
            del data[y][col]
        dic[key].append(data[y])
    #print(dic)    
    return attr, dic

def entropy(S):
    attr = list(set(S))
    if len(attr) == 1: 
        return 0
    counts = [0,0] 
    for i in range(2):
        counts[i] = sum([1 for x in S if attr[i] == x]) / (len(S) * 1.0)
        #print(counts) 
    sums = 0
    for cnt in counts:
        sums += -1 * cnt * math.log(cnt, 2)
    return sums

def compute_gain(data, col):   # dataset, 0
    attValues, dic = subtables(data, col, delete=False)
    total_entropy = entropy([row[-1] for row in data])
    for x in range(len(attValues)):
        ratio = len(dic[attValues[x]]) / ( len(data) * 1.0) #(|Sv|/|S|)
        entro = entropy([row[-1] for row in dic[attValues[x]]]) #Entropy of each instance in attribute
        total_entropy -= ratio*entro #(|Sv|/|S|)*Ev
    return total_entropy

def build_tree(data, features):
    lastcol = [row[-1] for row in data]  # get all the target values  
    if (len(set(lastcol))) == 1: # If all samples have same labels return that label
        node=Node("")
        node.answer = lastcol[0]
        return node
    n = len(data[0])-1
    #print("we are here",data[0])
    gains = [compute_gain(data, col) for col in range(n) ] #obtaining the list of gains for each column (i.e. attribute)
    split = gains.index(max(gains)) # Find max gains and returns index
    node = Node(features[split]) # 'node' stores attribute selected
    #del (features[split])
    fea = features[:split]+features[split+1:]
    attr, dic = subtables(data, split, delete=True) # Data will be spilt in subtables
    for x in range(len(attr)):
        child = build_tree(dic[attr[x]], fea)
        node.children.append((attr[x], child))
    return node

def print_tree(node, level):
    if node.answer != "":
        print(" "*level, node.answer) # Displays leaf node yes/no
        return
    print(" "*level, node.attribute) # Displays attribute Name
    for value, n in node.children: #n: The attribute that the label (value) leads to, value: attribute label
        print(" "*(level+1), value) 
        print_tree(n, level + 2)
    
def classify(node,x_test,features):
    if node.answer != "":
        print(node.answer)
        return
    pos = features.index(node.attribute)
    for value, n in node.children:
        if x_test[pos]==value:
            classify(n,x_test,features)
            
            
''' Main program '''
dataset, features = load_csv("/content/4train.csv") # change path of csv file
node = build_tree(dataset, features) # Build decision tree
print("The decision tree for the dataset using ID3 algorithm is ")
print_tree(node, 0)
testdata, features = load_csv("/content/4test.csv") #change path of csv file
for xtest in testdata:
    print("The test instance : ",xtest)
    print("The predicted label : ", end="")
    classify(node,xtest,features)
