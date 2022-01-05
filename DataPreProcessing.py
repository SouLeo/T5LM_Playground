import data_input
import t5_prompt_creation
import helper_functions


class DataPreProcessing:
    # TODO: Iron this out, not currently working
    def __init__(self):
        self.datainput = data_input.DataInput()
        self.training_prompts, self.training_jsons = self.datainput.get_training_data()
        self.valid_prompts, self.valid_jsons = self.datainput.get_valid_data()

        self.training_labels, self.valid_labels = self.collect_json_data()

        self.prompt_size = 100
        self.training_prompts, self.training_targets = self.datainput.get_training_data()

        self.inputs = self.create_prompt()

    def collect_json_data(self):
        return self.datainput.get_json_as_string(self.training_jsons), self.datainput.get_json_as_string(self.valid_jsons)

    def create_prompt(self):
        prompt_tokens, prompt_inits = t5_prompt_creation.create_prompt_tokens(self.prompt_size)
        return t5_prompt_creation.create_model_inputs(prompt_tokens, self.training_prompts)

    def verify_seq_len(self):
        max_seq_len = 512
        self.inputs = helper_functions.check_sequence_len(max_seq_len, self.inputs)
        self.training_labels = helper_functions.check_sequence_len(max_seq_len, self.training_labels)


