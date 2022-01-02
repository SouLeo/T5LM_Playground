import data_input


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('you are running big main')
    # Collecting Data
    datainput = data_input.DataInput()
    training_prompts, training_jsons = datainput.get_training_data()
    valid_prompts, valid_jsons = datainput.get_valid_data()
    # End Data Collection

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
