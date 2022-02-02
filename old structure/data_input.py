import os
import json


class DataInput:
    def __init__(self):
        self.file_path = os.getcwd() + '/sid-data'
        self.data = self.extract_umrf_data()
        # self.training_split = 251  # len(self.data)

        self.training_jsons = None
        self.training_jsons_str = None
        self.train_nl_prompts = None
        self.get_training_data()

    def get_nl_prompt(self, json_dicts: list):
        nl_prompts = []
        for json in json_dicts:
            nl_prompts.append(json['graph_description'])
        return nl_prompts

    def extract_umrf_data(self):
        json_list = [pos_json for pos_json in os.listdir(self.file_path) if pos_json.endswith('.json')]
        json_list_cropped = [e[11:] for e in json_list]
        json_list_cropped = [e[:-11] for e in json_list_cropped]
        json_cropped_int = [int(x) for x in json_list_cropped]
        json_cropped_int.sort()
        json_cropped_int = [str(x) for x in json_cropped_int]

        x = range(len(json_cropped_int))
        sid_file_names = []
        for i in x:
            sid_file_string = 'umrf_graph_' + json_cropped_int[i] + '.umrfg.json'
            sid_file_names.append(sid_file_string)

        json_dicts = []
        for json_file in sid_file_names:
            with open(self.file_path + '/' + json_file, 'r') as f:
                json_string = f.read()
                json_dict = json.loads(json_string)
                del json_dict['graph_name']
                del json_dict['graph_state']
                json_dicts.append(json_dict)
        return json_dicts

    def get_training_data(self):
        # self.training_jsons = self.data[:self.training_split]
        self.training_jsons = self.data
        self.train_nl_prompts = self.get_nl_prompt(self.training_jsons)
        labels = []
        for ex in self.training_jsons:
            items_list = []
            for items in ex['umrf_actions']:
                del items['id']
                del items['effect']
                del items['package_name']
                items_list.append(items)
            labels.append(items_list)
        self.training_jsons_str = self.get_json_as_string(labels)

    def get_json_as_string(self, json_list: list):
        json_strings = []
        for item in json_list:
            json_strings.append(json.dumps(item))
        return json_strings


if __name__ == '__main__':
    datainput = DataInput()
    print('you are running data input with file path: ' + datainput.file_path)
    training_prompts, training_jsons = datainput.get_training_data()
    print('data collection finished')
