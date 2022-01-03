import torch

import data_input
import t5_prompt_creation
import transformers


def check_sequence_len(max_len: int, inputs: list):
    # Method 1: Limit character len
    inputs = [e[:max_len] for e in inputs]
    # Method 2: Limit token len
    # i = 0
    # for item in inputs:
    #     num_tokens = len(item.split())
    #     print('old length ' + str(num_tokens))
    #     if num_tokens > max_len:
    #         cropped_input = item.split()[:max_len]
    #         print('new length ' + str(len(cropped_input)))
    #         item[i] = ' '.join(cropped_input)
    #     i = i + 1
    return inputs


def add_padding(examples: list):
    # The code below tests the max sequence length of my data.
    # The max is 254, so I manually set the padding to 255
    #
    # max_val = 0
    # for item in examples:
    #     if len(item) > max_val:
    #         max_val = len(item)
    #
    max_val = 255
    for ex in examples:
        ex.extend([0] * (max_val - len(ex)))
    return examples


if __name__ == '__main__':
    print('you are running big main')

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
    inputs = check_sequence_len(max_seq_len, inputs)
    training_labels = check_sequence_len(max_seq_len, training_labels)
    # end of sequence length checks

    # model setup
    tokenizer = transformers.T5Tokenizer.from_pretrained('t5-small')
    model = transformers.AutoModel.from_pretrained("google/t5-small-lm-adapt")
    inputs_encoded, attention_mask = tokenizer(inputs).data['input_ids'], tokenizer(inputs).data['attention_mask'],
    labels_encoded = torch.tensor(add_padding(tokenizer(training_labels).data['input_ids']))
    labels_encoded[labels_encoded == tokenizer.pad_token_id] = -100

    input_tensor = torch.tensor(inputs_encoded)
    loss = model(input_ids=input_tensor).loss
    print(loss)
    print('end')

