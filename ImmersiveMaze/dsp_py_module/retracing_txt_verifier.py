# Carol He (UCSB-Oct-2022):carol.hcxy@gmail.com

import re
from dsp_py_module.data_storage import DataStorage
from dsp_py_module.experiment_info import ExperimentInfo

REGEX_ID = re.compile(r".*Participant\s*ID\s*:\D*(\w+)")
REGEX_RELEARN_NUM = re.compile(r".*Re-learned\s*route\s*\D*(\d+)\s*times")

REGEX_SEQUENCE = [REGEX_ID, REGEX_RELEARN_NUM]
COLUMN_FIELD = ["id", "ReLearned"]

OUTPUT_COLUMN_ORDER = ["id", "ReLearned"]
OUTPUT_COLUMN_NAME_DICT = {"id": "ParticipantID",
                           "ReLearned": "ReLearnedRoute"}


class ReLearnInfo(ExperimentInfo):
    @staticmethod
    def factory(match, participant_id):
        relearned = match[1][0].strip()
        return ReLearnInfo(COLUMN_FIELD, [participant_id, relearned])


def relearn_generator(filename):
    with open(filename, 'r') as f:
        participant_id = 0
        current_matches = []
        cur_i = 0
        
        for line in f:
            for i in range(cur_i,len(REGEX_SEQUENCE)):
                match = REGEX_SEQUENCE[i].match(line)
                if match:
                    if participant_id == 0 and i == 0:
                        participant_id = match.group(1)
                        cur_i = 1
                    current_matches.append(match.groups())
                if i == len(REGEX_SEQUENCE) - 1 and match:
                    relearned = match[1][0].strip()
                    yield ReLearnInfo(COLUMN_FIELD, [participant_id, relearned])
                    cur_i = 0
                    current_matches = []
                    break
                

class RetracingTxtVerifier:

    def __init__(self, directory, suffix=".txt"):
        self.participant_id = []
        self.table = {}
        self.data = []
        self._files_paths = DataStorage.get_all_files_with_suffix(directory, suffix)

    def csv_generator(self):
        for file in self._files_paths:
            for row in relearn_generator(file):
                yield row.ordered_keys(OUTPUT_COLUMN_ORDER)

    def save(self, filename="txt_out.csv", column_name_dict=None, delimiter=","):
        if column_name_dict is None:
            column_name_dict = OUTPUT_COLUMN_NAME_DICT

        column_name = [OUTPUT_COLUMN_NAME_DICT[x] for x in OUTPUT_COLUMN_ORDER]

        rows = self.csv_generator()
        DataStorage.write_to_csv(filename, column_name, rows, delimiter)
