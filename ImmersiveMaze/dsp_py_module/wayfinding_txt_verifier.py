# rongfei@ucsb.edu carol.hcxy@gmail.com

import re
from dsp_py_module.data_storage import DataStorage
from dsp_py_module.experiment_info import ExperimentInfo

REGEX_ID = re.compile(r".*Participant\s*ID\s*:\D*(\w+)")
REGEX_TRIAL_NUMBER = re.compile(r".*No\D*(\d+).*Begins")
REGEX_STARTING_OBJECT = re.compile(r".*User.*confirmed.*on.*:(.*)[,|]")
REGEX_RESULT = re.compile(r".*Result.*:(.*)object:(.*)[,|].*Time\D*(\d+.*\d+)")
REGEX_END_MARK = re.compile(r".*No\D*(\d+).*Ends")

REGEX_SEQUENCE = [REGEX_ID, REGEX_TRIAL_NUMBER, REGEX_STARTING_OBJECT, REGEX_RESULT, REGEX_END_MARK]
COLUMN_FIELD = ["id", "order", "number", "starting", "target", "status", "time"]

OUTPUT_COLUMN_ORDER = ["id", "order", "number", "starting", "target", "status", "time"]
OUTPUT_COLUMN_NAME_DICT = {"id": "ParticipantID",
                           "order": "Order",
                           "number": "TrialNumber",
                           "starting": "StartingObject",
                           "target": "TargetObject",
                           "status": "Status",
                           "time": "Time"}


class TrialInfo(ExperimentInfo):
    @staticmethod
    def check_status(status):
        if status == "Success":
            return status
        return "Failure"

    @staticmethod
    def factory(match, participant_id, order):
        trial_number = match[0][0].strip()
        starting = match[1][0].strip()
        status = TrialInfo.check_status((match[2][0].strip()))
        target = match[2][1].strip()
        time = match[2][2].strip()
        return TrialInfo(COLUMN_FIELD, [participant_id, order, trial_number, starting, target, status, time])


def trial_generator(filename):
    with open(filename, 'r') as f:
        order = 0
        participant_id = 0
        current_matches = []
        cur_i = 0
        
        for line in f:
            for i in range(cur_i,len(REGEX_SEQUENCE)):
                match = REGEX_SEQUENCE[i].match(line)
                if match:
                    if i == 0:
                        participant_id = match.group(1)
                        cur_i = 1
                        break
                    current_matches.append(match.groups())
                    cur_i = len(current_matches)+1
                if i == len(REGEX_SEQUENCE) - 1 and match:
                    order += 1
                    cur_i = 1
                    yield TrialInfo.factory(current_matches, participant_id, order)
                    current_matches = []
                


class WayFindingTxtVerifier:

    def __init__(self, directory, suffix=".txt"):
        self.participant_id = []
        self.table = {}
        self.data = []
        self._files_paths = DataStorage.get_all_files_with_suffix(directory, suffix)

    def csv_generator(self):
        for file in self._files_paths:
            for row in trial_generator(file):
                yield row.ordered_keys(OUTPUT_COLUMN_ORDER)

    def save(self, filename="txt_out.csv", column_name_dict=None, delimiter=","):
        if column_name_dict is None:
            column_name_dict = OUTPUT_COLUMN_NAME_DICT

        column_name = [OUTPUT_COLUMN_NAME_DICT[x] for x in OUTPUT_COLUMN_ORDER]

        rows = self.csv_generator()
        DataStorage.write_to_csv(filename, column_name, rows, delimiter)
