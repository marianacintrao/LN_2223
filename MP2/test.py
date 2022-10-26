import pickle
import naive_bayes_classifier
#import rule_based_classifier
import data_prep
import json

results = {}
data = []
target = []
target_names = [
    "==Poor==", 
    "==Unsatisfactory==", 
    "==Good==", 
    "==VeryGood==", 
    "==Excellent=="
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
    
    '''
    #########################
    # test with naive bayes #
    #########################
    if "naive_bayes" not in results.keys():
        results["naive_bayes"] = []
    nb_predicted_data = naive_bayes_classifier.classify(train_data, train_target, target_names, test_data)
    iteration_results = {"correct": 0, "incorrect": 0, "incorrect_instances": []}
    print("Comparing predicted and actual data")
    for j in range(len(nb_predicted_data)):
        text, predicted_category = nb_predicted_data[j]
        target_category_code = test_target[j]
        target_category = target_names[target_category_code]
        if predicted_category != target_category:
            iteration_results["incorrect"] += 1
            iteration_results["incorrect_instances"].append({
                "text": text, 
                "predicted": predicted_category, 
                "target": target_category, 
                "error_distance": target_category_code - target_names.index(predicted_category)})
        else:
            iteration_results["correct"] += 1

    results["naive_bayes"].append(iteration_results)

    '''
    ############################################
    # test with naive bayes and pre processing #
    ############################################
    
    # def transform_data(data, REMOVE_STOPWORDS=False, PORTERSTEMMER=False, 
    #                          LANCASTERSTEMMER=False, LEMMATIZATION=False):

    for args in ( 
        (False, False, False, False), (False, True, False, False), 
        (False, False, True, False), (False, False, False, True),
        (True, False, False, False), (True, True, False, False), 
        (True, False, True, False), (True, False, False, True),
        ):
        iteration_results = {"correct": 0, "incorrect": 0, "incorrect_instances": []}
        strategy_name = "naive_bayes_and_"+str(args)
        if strategy_name not in results.keys():
            results[strategy_name] = []

        train_data = data_prep.transform_data(train_data, *args)
        test_data = data_prep.transform_data(test_data, *args)

        nb_predicted_data = naive_bayes_classifier.classify(train_data, train_target, target_names, test_data)
        iteration_results = {"correct": 0, "incorrect": 0, "incorrect_instances": []}
        print("Comparing predicted and actual data")
        for j in range(len(nb_predicted_data)):
            text, predicted_category = nb_predicted_data[j]
            target_category_code = test_target[j]
            target_category = target_names[target_category_code]
            if predicted_category != target_category:
                iteration_results["incorrect"] += 1
                iteration_results["incorrect_instances"].append({
                    "text": text, 
                    "predicted": predicted_category, 
                    "target": target_category, 
                    "error_distance": target_category_code - target_names.index(predicted_category)})
            else:
                iteration_results["correct"] += 1

        results[strategy_name].append(iteration_results)
    

    #####################
    # other tests: TODO #
    #####################



print("Printing results to file")
jsondump = json.dumps(results, indent=4)
# Writing to results.json
with open("results.json", "w") as outfile:
    outfile.write(jsondump)



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
    