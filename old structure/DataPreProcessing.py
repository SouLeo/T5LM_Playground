import data_input
import t5_prompt_creation


class DataPreProcessing:
    def __init__(self, prompt_size):
        self.prompt_size = prompt_size
        self.datainput = data_input.DataInput()
        self.prompt_tokens, self.prompt_inits = t5_prompt_creation.create_prompt_tokens(self.prompt_size)

        self.training_input = t5_prompt_creation.create_model_inputs(self.prompt_tokens, self.datainput.train_nl_prompts)
        self.training_labels = self.datainput.training_jsons_str

    def get_prompt_tokens(self):
        return self.prompt_tokens
