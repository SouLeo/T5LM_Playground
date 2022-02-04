import sys
import os
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import transformers
from DataPreProcessing import DataPreProcessing
import helper_functions


# Checks system for gpu availability
def check_for_gpu():
    # checking if gpu can be used in training
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('gpu found')
    else:
        device = torch.device('cpu')
        print('using cpu')
    return device


if __name__ == '__main__':
    # Example Command:
    # Python3 main.py 1500 t5-small /saved_model /saved_tokenizer
    args = sys.argv
    model_name = args[1]  # t5-small, t5-base, t5-large
    token_save_name = args[2]
    prompt_size = int(args[3])  # 50

    print('you are running the prediction program')
    device = check_for_gpu()

    data_preprocess = DataPreProcessing(prompt_size)

    # Required for memory constraints of T5-small?
    max_seq_len = 4096  # Design constraint for t5-small model
    training_inputs = helper_functions.check_sequence_len(max_seq_len, data_preprocess.training_input)
    training_labels = helper_functions.check_sequence_len(max_seq_len, data_preprocess.training_labels)

    # Ensure Prompts are only Tuned
    vocab = range(3218)
    prompt_token_indices = range(prompt_size)
    prompt_token_indices = [x + 32100 for x in prompt_token_indices]
    mask = list(set(vocab) ^ set(prompt_token_indices))
    # End of Prompt Tuning Enforcement

    # model setup
    tokenizer = transformers.T5Tokenizer.from_pretrained(os.getcwd()+token_save_name)
    model = transformers.T5ForConditionalGeneration.from_pretrained(model_name).to(device)
    model.resize_token_embeddings(len(tokenizer))
    optimizer = transformers.AdamW(model.parameters(), lr=0.001)
    checkpoint = torch.load(os.getcwd() + '/model_checkpoint-'+model_name)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(checkpoint['epoch'])

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    training_data_loader = torch.load('training_data_loader-' + model_name)
    valid_data_loader = torch.load('valid_data_loader-' + model_name)

    if device.type == 'cuda':
        print('model parallelization on gpu')
        model.parallelize()

    # Inference from Validation
    print('beginning inference')
    predictions = []
    ground_truths = []
    for batch in valid_data_loader:
        generated_ids = model.generate(batch[0], max_length=1000)
        pred_json_labels = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for preds in pred_json_labels:
            predictions.append(preds)
        truth_labels_debug = batch[1]
        truth_labels_debug[truth_labels_debug == -100] = tokenizer.pad_token_id
        truth_labels = tokenizer.batch_decode(truth_labels_debug, skip_special_tokens=True)
        for label in truth_labels:
            ground_truths.append(label)

    pred_file = open('preds-1-' + model_name + '.txt', 'w')
    for ex in predictions:
        pred_file.write(ex + '\n')
    pred_file.close()

    label_file = open('truths-1-' + model_name + '.txt', 'w')
    for ex in ground_truths:
        label_file.write(ex + '\n')
    label_file.close()

    print('end')
