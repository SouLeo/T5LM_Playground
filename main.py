import data_input
import t5_prompt_creation
import transformers


if __name__ == '__main__':
    print('you are running big main')
    # Collecting Data
    datainput = data_input.DataInput()
    training_prompts, training_jsons = datainput.get_training_data()
    valid_prompts, valid_jsons = datainput.get_valid_data()
    # End Data Collection

    # Create and init prompts
    prompt_size = 100
    prompt_tokens, prompt_inits = t5_prompt_creation.create_prompt_tokens(prompt_size)
    inputs = t5_prompt_creation.create_model_inputs(prompt_tokens, training_prompts)
    # End prompt creation and init

    # model = transformers.AutoModel.from_pretrained("google/t5-small-lm-adapt")
    print('end')

