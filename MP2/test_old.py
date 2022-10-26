import pickle
import naive_bayes_classifier
import rule_based_classifier
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
                "actual": target_category, 
                "error_distance": target_category_code - target_names.index(predicted_category)})
        else:
            iteration_results["correct"] += 1

    results["naive_bayes"].append(iteration_results)

    ###################
    # test with rules #
    ###################
    if "rule_based" not in results.keys():
        results["rule_based"] = []

    rb_predicted_data = rule_based_classifier.classify(test_data)
    iteration_results = {"correct": 0, "incorrect": 0, "incorrect_instances": []}
    print("Comparing predicted and actual data")
    for j in range(len(rb_predicted_data)):
        text, predicted_category = rb_predicted_data[j]
        target_category_code = test_target[j]
        target_category = target_names[target_category_code]
        if predicted_category != target_category:
            iteration_results["incorrect"] += 1
            iteration_results["incorrect_instances"].append({
                "text": text, 
                "predicted": predicted_category, 
                "actual": target_category, 
                "error_distance": target_category_code - target_names.index(predicted_category)})
        else:
            iteration_results["correct"] += 1

    results["rule_based"].append(iteration_results)
        
    ###################################
    # test with rules and naive bayes #
    ###################################

    rb_predicted_data = rule_based_classifier.classify(test_data, True, False)

    #rb_weight = .3
    for rb_weight in [.2, .3, .4, .5]:
        iteration_results = {"correct": 0, "incorrect": 0, "incorrect_instances": []}
        strategy_name = "naive_bayes_and_rule_based_wo_stopwords_"+str(rb_weight)

        if strategy_name not in results.keys():
            results[strategy_name] = []

        print("Comparing predicted and actual data")
        for k in range(len(nb_predicted_data)):
            nb_text, nb_predicted_category = nb_predicted_data[k]
            rb_text, rb_predicted_category = rb_predicted_data[k]

            nb_predicted_category_code = target_names.index(nb_predicted_category)
            rb_predicted_category_code = target_names.index(rb_predicted_category)

            predicted_category_code = round((nb_predicted_category_code * (1-rb_weight)) + (rb_predicted_category_code * rb_weight))
            predicted_category = target_names[predicted_category_code]

            target_category_code = test_target[k]
            target_category = target_names[target_category_code]

            if predicted_category != target_category:
                iteration_results["incorrect"] += 1
                iteration_results["incorrect_instances"].append({
                    "text": nb_text, 
                    "predicted": predicted_category, 
                    "actual": target_category, 
                    "error_distance": target_category_code - target_names.index(predicted_category)})
            else:
                iteration_results["correct"] += 1

        results[strategy_name].append(iteration_results)
    
    #############################
    # test with rules and stuff #
    #############################
    # def classify(data, STOP_WORD_REMOVAL=False, AUTOCORRECT=False):
    for args in ((test_data, False, False), (test_data, True, False)):
        #, (test_data, False, True), (test_data, True, True)):
        iteration_results = {"correct": 0, "incorrect": 0, "incorrect_instances": []}
        strategy_name = "rule_based_"+str(args[1:])
        if strategy_name not in results.keys():
            results[strategy_name] = []

        rb_predicted_data = rule_based_classifier.classify(*args)
        iteration_results = {"correct": 0, "incorrect": 0, "incorrect_instances": []}
        print("Comparing predicted and actual data")
        for j in range(len(rb_predicted_data)):
            text, predicted_category = rb_predicted_data[j]
            target_category_code = test_target[j]
            target_category = target_names[target_category_code]
            if predicted_category != target_category:
                iteration_results["incorrect"] += 1
                iteration_results["incorrect_instances"].append({
                    "text": text, 
                    "predicted": predicted_category, 
                    "actual": target_category, 
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
    