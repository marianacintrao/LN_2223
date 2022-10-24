import pickle
import naive_bayes_classifier
import json

results = {"naive_bayes": []}
data = []
target = []
target_names = [
    "=Poor=", 
    "=Unsatisfactory=", 
    "=Good=", 
    "=VeryGood=", 
    "=Excellent="
    ]

for i in range(10):
    # load dict from file with data
    train_file = open("split_datasets/train_"+str(i)+".pkl", "rb")
    train_dict = pickle.load(train_file)
    data.append(train_dict["data"])
    target.append(train_dict["target"])


for i in range(10):
    print("Testing with train set", i)
    train_data = [item for sublist in data[:i]+data[i+1:] for item in sublist]
    train_target = [item for sublist in target[:i]+target[i+1:] for item in sublist]
    test_data = data[i]
    test_target = target[i]

    #########################
    # test with naive bayes #
    #########################
    predicted_data = naive_bayes_classifier.classify(train_data, train_target, target_names, test_data)
    iteration_results = {"correct": 0, "incorrect": 0, "incorrect_instances": []}
    print("Comparing predicted and actual data")
    for j in range(len(predicted_data)):
        text, category = predicted_data[j]
        predicted_category_code = test_target[j]
        predicted_category = target_names[predicted_category_code]
        if predicted_category != category:
            iteration_results["incorrect"] += 1
            iteration_results["incorrect_instances"].append({
                "text": text, 
                "predicted": category, 
                "actual": predicted_category, 
                "error_distance": predicted_category_code - target_names.index(category)})
        else:
            iteration_results["correct"] += 1

    results["naive_bayes"].append(iteration_results)
    #####################
    # other tests: TODO #
    #####################


print("Printing results to file")
j = json.dumps(results, indent=4)
# Writing to results.json
with open("results.json", "w") as outfile:
    outfile.write(j)



print("Getting stats!")
for key in results.keys():
    print("="*len(key), "\n"+key, "\n"+"="*len(key))
    print("  Correct values per iteration:")
    print("   ", [iteration_results["correct"] for iteration_results in results[key]])
    average_correct_percentage = sum([iteration_results["correct"] for iteration_results in results[key]]) / (1000*10)
    print(" ", "Average correct classification percentage:", "\n    {0:.2%}".format(average_correct_percentage))


    items = [item for iteration in results[key] for item in iteration["incorrect_instances"]]
    distances = [item["error_distance"] for item in items]
    print("  Error stats:")
    print("    Average:", "\n     ", sum(distances)/len(distances))
    dist_set = list(set(distances))
    dist_occ = {}
    for dist in dist_set:
        dist_occ[dist] = distances.count(dist)
    print("    Distance occurences:","\n     ",  dist_occ)
    #print("    Problematic sentences: ")
    problematic_items = {}
    #for item in items:
        #if abs(item["error_distance"]) > 3:
        #    print("     ", item["text"], "|| pred:", item["predicted"], "|| actual:", item["actual"])
    