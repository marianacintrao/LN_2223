import pickle
import collections

train_file = open("train.pkl", "rb")
train = pickle.load(train_file)

data = train["data"]
target = train["target"]

repeated_items = [item for item, count in collections.Counter(data).items() if count > 1]

target_names = [
    "=Poor=", 
    "=Unsatisfactory=", 
    "=Good=", 
    "=VeryGood=", 
    "=Excellent="
    ]

for item in repeated_items:
    i = 0
    class_list = []
    for _ in range(data.count(item)):
        index = data[i:].index(item)
        class_list.append(target_names[target[index + i]])
        i += index + 1
    print("\n", item, "\n", class_list) 

print("\n", "=" * 20)

for i in range(len(data)):
    if not any(c.isalpha() for c in data[i]):
        print(data[i], target_names[target[i]])