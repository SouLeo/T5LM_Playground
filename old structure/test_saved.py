import transformers
import t5_prompt_creation

if __name__ == '__main__':
    # Inference from Validation

    model = transformers.T5ForConditionalGeneration.from_pretrained('./saved_model')
    model.eval()
    tokenizer = transformers.T5Tokenizer.from_pretrained('./saved_tokenizer')

    ex = ['go to the desk']
    inputs = t5_prompt_creation.ensemble_prompt_creation(10, ex)
    input_ids = tokenizer.encode(inputs[0], max_length=512, padding='longest', truncation=True, return_tensors='pt').data
    generated_ids = model.generate(input_ids, max_length=200)
    test = generated_ids.squeeze()
    pred_json_labels = tokenizer.decode(test, skip_special_tokens=True)
    prepend_str = '{ "umrf_actions": '
    postpend_str = '}'
    output = prepend_str + pred_json_labels + postpend_str
    print(output)

