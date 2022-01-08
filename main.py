import torch
from torch.utils.data import TensorDataset, DataLoader
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


def create_data_loader(tokenizer, inputs, labels):
    max_source_length = 512
    max_target_length = 256
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding="longest", truncation=True,
                             return_tensors="pt").data

    input_ids = model_inputs['input_ids'].to(device)
    labels_encoded = torch.tensor(
        tokenizer(labels, max_length=max_target_length, padding="longest", truncation=True).data[
            'input_ids']).to(device)

    labels_encoded[labels_encoded == tokenizer.pad_token_id] = -100

    training_dataset = TensorDataset(input_ids, labels_encoded)
    return DataLoader(training_dataset, shuffle=True, batch_size=8)


# TODO: Clean up this code and play with model input sizing for exs  and labels
if __name__ == '__main__':
    print('you are running the training program')
    device = check_for_gpu()

    data_preprocess = DataPreProcessing()

    # Required for memory constraints of T5-small?
    max_seq_len = 4096  # Design constraint for t5-small model
    training_inputs = helper_functions.check_sequence_len(max_seq_len, data_preprocess.training_input)
    training_labels = helper_functions.check_sequence_len(max_seq_len, data_preprocess.training_labels)

    valid_inputs = helper_functions.check_sequence_len(max_seq_len, data_preprocess.valid_input)
    valid_labels = helper_functions.check_sequence_len(max_seq_len, data_preprocess.valid_labels)
    # end of sequence length checks

    # model setup
    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small')
    punct_tokens = ['{', '}']
    tokenizer.add_tokens(punct_tokens)
    # model = transformers.AutoModel.from_pretrained("google/t5-small-lm-adapt")
    model = transformers.T5ForConditionalGeneration.from_pretrained('t5-small').to(device)
    # print(model.config)
    if device.type == 'cuda':
        # TODO: Ensure the parallelization is actually being utilized during model.generate on TACC
        print('model parallelization on gpu')
        model.parallelize()
    optimizer = transformers.AdamW(model.parameters(), lr=0.001)

    # create data loaders
    training_data_loader = create_data_loader(tokenizer, training_inputs, training_labels)
    valid_data_loader = create_data_loader(tokenizer, valid_inputs, valid_labels)

    # Debugging
    # ground_truths = []
    # for batch in training_data_loader:
    #     # Try to convert actual labels from id's to tokens
    #     truth_labels_debug = batch[1]
    #     truth_labels_debug[truth_labels_debug == -100] = tokenizer.pad_token_id
    #     truth_labels = tokenizer.batch_decode(truth_labels_debug, skip_special_tokens=True)
    #     for label in truth_labels:
    #         ground_truths.append(label)
    #     print('hi')
    # End debugging

    # Training Loop
    max_epochs = 10  # 6 or 9 or 10 best
    total_epochs = range(max_epochs)
    for epoch in total_epochs:
        model.train()
        for batch in training_data_loader:
            optimizer.zero_grad()
            out = model(input_ids=batch[0], labels=batch[1])
            loss = out.loss
            loss.backward()
            optimizer.step()

        # Validation per training
        losses = []
        model.eval()
        for batch in valid_data_loader:
            with torch.no_grad():
                outputs = model(input_ids=batch[0], labels=batch[1])
            losses.append(outputs.loss.item())
        losses = torch.FloatTensor(losses)
        avg_loss = torch.mean(losses)
        print('loss: ', avg_loss)

    # Inference from Validation
    print('beginning inference')
    predictions = []
    ground_truths = []
    for batch in valid_data_loader:
        generated_ids = model.generate(batch[0], max_length=1000)
        pred_json_labels = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        for preds in pred_json_labels:
            predictions.append(preds)
        # Try to convert actual labels from id's to tokens
        truth_labels_debug = batch[1]
        truth_labels_debug[truth_labels_debug == -100] = tokenizer.pad_token_id
        truth_labels = tokenizer.batch_decode(truth_labels_debug, skip_special_tokens=True)
        for label in truth_labels:
            ground_truths.append(label)

    pred_file = open('preds.txt', 'w')
    for ex in predictions:
        pred_file.write(ex + '\n')
    pred_file.close()

    label_file = open('truths.txt', 'w')
    for ex in ground_truths:
        label_file.write(ex + '\n')
    label_file.close()

    print('end')
