import random


def create_model_inputs(prompt_tokens: list, training_prompts: list):
    prompt_tokens_as_str = ' '.join(prompt_tokens)
    model_inputs = []
    for i in range(len(training_prompts)):
        model_inputs.append(prompt_tokens_as_str + ' ' + training_prompts[i])

    return model_inputs


def create_prompt_tokens(prompt_size: int):
    # using random initialization
    prompt_tokens = []
    for i in range(0, prompt_size):
        prompt_tokens.append('PROMPT_TOKEN' + str(i))

    prompt_token_initializations = []
    for i in range(0, prompt_size):
        n = random.uniform(-0.5, 0.5)
        prompt_token_initializations.append(n)

    return prompt_tokens, prompt_token_initializations


def ensemble_prompt_creation(prompt_size: int, training_prompts):
    prompt_tokens, prompt_inits = create_prompt_tokens(prompt_size)
    inputs = create_model_inputs(prompt_tokens, training_prompts)
    return inputs

if __name__ == '__main__':
    # TODO: Create prompt class for different initialization strategies
    print('you are running prompt tuning')

