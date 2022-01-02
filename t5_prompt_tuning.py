import transformers


if __name__ == '__main__':
    print('you are running prompt tuning')

    model = transformers.AutoModel.from_pretrained("google/t5-small-lm-adapt")
