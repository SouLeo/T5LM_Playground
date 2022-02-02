import json
from nltk.translate.bleu_score import sentence_bleu
from nltk.tokenize import sent_tokenize, word_tokenize
from statistics import mean, median

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
    #
    # # online examples
    # str = truths[1]
    # split = str.split()
    # test_ref = [split]
    # print(test_ref)
    # test_ex = preds[1]
    # test_str = test_ex.split()
    # print(test_str)
    # score = sentence_bleu(test_ref, test_str)
    # print(score)
    #
    # reference = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
    # candidate = ['this', 'is', 'a', 'test']
    # score = sentence_bleu(reference, candidate)
    # print(score)
    # # end online examples


    ## BEGIN BLEU EVALUATION
    bleu_scores = []
    for i in range(len(truths)):
        label = truths[i].split()
        ref = [label]
        pred_bleu = preds[i].split()
        bleu_score = sentence_bleu(ref, pred_bleu)
        bleu_scores.append(bleu_score)
    ## END BLEU EVALUATION
    print('median score: ', median(bleu_scores))
    print('average score: ', mean(bleu_scores))

    # truth_jsons = []
    # pred_jsons = []
    # for i in range(len(truths)):
    #     try:
    #         true_json = json.loads(truths[i])
    #         pred_json = json.loads(preds[i])
    #
    #         truth_jsons.append(true_json)
    #         pred_jsons.append(pred_json)
    #     except:
    #         truth_jsons.append('nan')
    #         pred_jsons.append('nan')
    #         # print('issues converting examples to jsons')

    # uncomment below
    # invalid_pred_jsons = []
    # corresponding_labels_json = []
    # truth_jsons = []
    # pred_jsons = []
    # for i in range(len(truths)):
    #     try:
    #         true_json = json.loads(truths[i])
    #         truth_jsons.append(true_json)
    #     except:
    #         truth_jsons.append('nan')
    #         pred_jsons.append('nan')
    #         continue
    #     try:
    #         pred_json = json.loads(preds[i])
    #         pred_jsons.append(pred_json)
    #     except:
    #         pred_jsons.append('nan')
    #         invalid_pred_jsons.append(preds[i])
    #         corresponding_labels_json.append(truths[i])
    #
    #
    #         # print('issues converting examples to jsons')
    #
    # correct_names = []
    # correct_description = []
    # correct_input_params = []
    # num_of_invalid_jsons = []
    # for i in range(len(truth_jsons)):
    #     if truth_jsons[i] != 'nan':
    #         if pred_jsons[i] == 'nan':
    #             num_of_invalid_jsons.append(1)
    #             continue
    #         else:
    #             num_of_invalid_jsons.append(0)
    #         label = truth_jsons[i]
    #         pred = pred_jsons[i]
    #         for j in range(len(label)):
    #             label_action = label[j]
    #             pred_action = pred[j]
    #             # Check if names are equal
    #             if label_action['name'] == pred_action['name']:
    #                 correct_names.append(1)
    #             else:
    #                 correct_names.append(0)
    #             # check if descriptions are equal
    #             label_stand_desc = ''.join(e for e in label_action['description'] if e.isalnum())
    #             pred_stand_desc = ''.join(e for e in pred_action['description'] if e.isalnum())
    #             if label_stand_desc == pred_stand_desc:
    #                 correct_description.append(1)
    #             else:
    #                 correct_description.append(0)
    #             # check input params
    #             label_input_param_keys = label_action['input_parameters'].keys()
    #             pred_input_param_keys = pred_action['input_parameters'].keys()
    #             if label_input_param_keys == pred_input_param_keys:
    #                 for item in label_input_param_keys:
    #                     a = ''.join(e for e in label_action['input_parameters'][item]['pvf_example'] if e.isalnum())
    #                     b = ''.join(e for e in pred_action['input_parameters'][item]['pvf_example'] if e.isalnum())
    #                     if a == b:
    #                         correct_input_params.append(1)
    #                     else:
    #                         correct_input_params.append(0)
    #             else:
    #                 print('input keys misaligned or do not match')
    #
    # # Coarse accuracy begins
    # num_exs = len(correct_description)
    # mean_correct_description = mean(correct_description)
    # mean_correct_input_params = mean(correct_input_params)
    # mean_correct_names = mean(correct_names)
    # # Coarse accuracy ends
    # print(str(num_exs))
    # print('avg description correct: ', mean_correct_description)
    # print('avg input params correct: ', mean_correct_input_params)
    # print('avg names correct: ', mean_correct_names)
    #
    # total = sum(num_of_invalid_jsons)
    # print('num invalid pred jsons: ', total)
    # print('invalid pred jsons:')
    # for i in range(total):
    #     print('prediction: ', invalid_pred_jsons[i])
    #     print('label: ', corresponding_labels_json[i])
    #     print('\n')
    print('end')

