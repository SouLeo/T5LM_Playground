

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
    max_val = 280
    for ex in examples:
        ex.extend([0] * (max_val - len(ex)))
    return examples
