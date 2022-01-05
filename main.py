import torch
from torch.utils.data import TensorDataset, DataLoader
import transformers
from DataPreProcessing import DataPreProcessing
import data_input
import t5_prompt_creation
import helper_functions


if __name__ == '__main__':
    print('you are running the training program')

    # checking if gpu can be used in training
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print('gpu found')
    else:
        device = torch.device('cpu')
        print('using cpu')
#    device = torch.device('cpu')

    # Collecting Data
    datainput = data_input.DataInput()
    training_prompts, training_jsons = datainput.get_training_data()
    valid_prompts, valid_jsons = datainput.get_valid_data()

    training_labels = datainput.get_json_as_string(training_jsons)
    valid_labels = datainput.get_json_as_string(valid_jsons)
    # End Data Collection

    # Create and initialize prompts
    prompt_size = 100
    prompt_tokens, prompt_inits = t5_prompt_creation.create_prompt_tokens(prompt_size)
    inputs = t5_prompt_creation.create_model_inputs(prompt_tokens, training_prompts)
    # End prompt creation and init

    # verifying sequence lengths are in good standing w/ model constraints
    max_seq_len = 512  # Design constraint for t5-small model
    inputs = helper_functions.check_sequence_len(max_seq_len, inputs)
    training_labels = helper_functions.check_sequence_len(max_seq_len, training_labels)
    # end of sequence length checks

    # preprocessor = DataPreProcessing()
    # inputs = preprocessor.inputs
    # targets = preprocessor.training_labels

    # model setup
    max_source_length = 512
    max_target_length = 128

    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small')
    model = transformers.T5ForConditionalGeneration.from_pretrained('t5-small')
    optimizer = transformers.AdamW(model.parameters(), lr=0.001)
    # model = transformers.AutoModel.from_pretrained("google/t5-small-lm-adapt")
    model = model.to(device)
    model.parallelize()
    model_inputs = tokenizer(inputs, max_length=max_source_length, padding="longest", truncation=True, return_tensors="pt").data
    input_ids = model_inputs['input_ids'].to(device)
    attention_mask = model_inputs['attention_mask'].to(device)

    labels_encoded = torch.tensor(tokenizer(training_labels, max_length=max_target_length, padding="longest", truncation=True).data['input_ids']).to(device)
    labels_encoded[labels_encoded == tokenizer.pad_token_id] = -100

    training_dataset = TensorDataset(input_ids, attention_mask, labels_encoded)
    training_data_loader = DataLoader(training_dataset, shuffle=True, batch_size=8)

    max_epochs = 1
    total_epochs = range(max_epochs)
    for epoch in total_epochs:
        model.train()
        for batch in training_data_loader:
            optimizer.zero_grad()
            outputs = model(input_ids=batch[0], attention_mask=batch[1], decoder_input_ids=batch[2])
            loss = outputs.loss
            print('loss ', loss.item())

            optimizer.step()
    print('end')
