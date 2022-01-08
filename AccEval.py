import json

if __name__ == '__main__':
    # Read in data inputs
    truths = []
    with open('truths.txt', 'r') as truth_file:
        for line in truth_file:
            truths.append(line.strip())

    preds = []
    with open('preds.txt', 'r') as pred_file:
        for line in pred_file:
            preds.append(line.strip())

    # Coarse accuracy begins
    # score = 0
    # for i in range(len(truth_jsons)):
    #     if truth_jsons[i] == pred_jsons[i]:
    #         score = score + 1
    #     else:
    #         print(truth_jsons[i])
    #         print(pred_jsons[i])
    # print('coarse acc is ' + str(score) + '/42')
    # Coarse accuracy ends


    # TODO: 
    # 1) remove broken ground_truth labels
    # 2) remove broken pred_labels
    # 3) calculate coarse accuracy
    # 4) calculate accuracy by jsons

    print('end')

