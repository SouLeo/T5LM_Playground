import transformers
import t5_prompt_creation

if __name__ == '__main__':
    # Inference from Validation

    model = transformers.T5ForConditionalGeneration.from_pretrained('./saved_model')
    model.eval()
    tokenizer = transformers.T5Tokenizer.from_pretrained('./saved_tokenizer')

    ex = ['Place the candle on top of the toilet']
    inputs = t5_prompt_creation.ensemble_prompt_creation(10, ex)
    input_ids = tokenizer.encode(inputs[0], max_length=512, padding='longest', truncation=True, return_tensors='pt').data
    generated_ids = model.generate(input_ids, max_length=1000)
    test = generated_ids.squeeze()
    pred_json_labels = tokenizer.decode(test, skip_special_tokens=True)
    print(pred_json_labels)

