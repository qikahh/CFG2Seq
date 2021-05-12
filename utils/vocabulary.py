import pickle
import json
from dataclasses import dataclass
from os.path import exists
from typing import Dict, Optional


# vocabulary keys
TOKEN_TO_ID = "token_to_id"
LABEL_TO_ID = "label_to_id"
ID_TO_TOKEN = "id_to_token"
ID_TO_LABEL = "id_to_label"
TYPE_TO_ID = "type_to_id"


# sequence service tokens
SOS = "<SOS>"
EOS = "<EOS>"
PAD = "<PAD>"
UNK = "<UNK>"
CLS = "<CLS>"
SEP = "<SEP>"
DIG = "<DIG>"


@dataclass
class Vocabulary:
    token_to_id: Dict[str, int]
    id_to_token: Dict[str, int]
    label_to_id: Dict[str, int]
    id_to_label: Dict[str, int]
    type_to_id: Optional[Dict[str, int]] = None

    @staticmethod
    def load_vocabulary(vocabulary_path: str) -> "Vocabulary":
        if not exists(vocabulary_path):
            raise ValueError(f"Can't find vocabulary in: {vocabulary_path}")
        with open(vocabulary_path, "r") as vocabulary_file:
            vocabulary_dicts = json.load(vocabulary_file) 
        token_to_id = vocabulary_dicts[TOKEN_TO_ID]
        label_to_id = vocabulary_dicts[LABEL_TO_ID]
        id_to_label = dict([int(val),key] for key,val in vocabulary_dicts[LABEL_TO_ID].items())
        id_to_token = dict([int(val),key] for key,val in vocabulary_dicts[TOKEN_TO_ID].items())
        type_to_id = vocabulary_dicts.get(TYPE_TO_ID, None)
        return Vocabulary(
            token_to_id=token_to_id, label_to_id=label_to_id, id_to_label=id_to_label, id_to_token=id_to_token, type_to_id=type_to_id
        )

    @staticmethod
    def dump_vocabulary(vacb, vocabulary_path: str):
        with open(vocabulary_path, "w", encoding='utf-8') as vocabulary_file:
            vocabulary_dicts = {
                TOKEN_TO_ID: vacb[TOKEN_TO_ID],
                LABEL_TO_ID: vacb[LABEL_TO_ID],
            }
            if vacb[TYPE_TO_ID] is not None:
                vocabulary_dicts[TYPE_TO_ID] = vacb[TYPE_TO_ID]
            json_data = json.dumps(vocabulary_dicts,indent=2,ensure_ascii=False)
            vocabulary_file.write(json_data)
