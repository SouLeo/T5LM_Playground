import torch
import transformers
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

    # model setup
    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small')
    model = transformers.AutoModel.from_pretrained("google/t5-small-lm-adapt")
    model = model.to(device)
    model.parallelize()
    inputs_encoded, attention_mask = torch.tensor(tokenizer(inputs).data['input_ids']).to(device), torch.tensor(tokenizer(inputs).data['attention_mask']).to(device),
    labels_encoded = torch.tensor(helper_functions.add_padding(tokenizer(training_labels).data['input_ids']))
    labels_encoded[labels_encoded == tokenizer.pad_token_id] = -100

    loss = model(input_ids=inputs_encoded, decoder_input_ids=labels_encoded).loss
    print(loss)
    print('end')
