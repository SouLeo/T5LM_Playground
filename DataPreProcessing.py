import data_input
import t5_prompt_creation


class DataPreProcessing:
    def __init__(self):
        self.datainput = data_input.DataInput()
        self.prompt_tokens, self.prompt_inits = t5_prompt_creation.create_prompt_tokens(prompt_size=100)

        self.training_input = t5_prompt_creation.create_model_inputs(self.prompt_tokens, self.datainput.train_nl_prompts)
        self.training_labels = self.datainput.training_jsons_str

        self.valid_input = t5_prompt_creation.create_model_inputs(self.prompt_tokens, self.datainput.valid_nl_prompts)
        self.valid_labels = self.datainput.valid_jsons_str

