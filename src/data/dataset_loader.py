import json
from pathlib import Path

import datasets

from misc.constant import DataType


DATASET_INFOS = {
    "hotpotqa": ["train", "dev"], 
    "2wiki": ["train", "dev"], 
    "musique": ["train", "dev"], 
    "bamboogle": ["test"],
    "aime24": ["train"],
    "aime25": ["test"],
    "math500": ["test"],
    "medqa": ["train", "dev", "test"],
    "gpqa": ["train"],
    "webqsp": ["train", "validation", "test"],
    "wtq": ["train", "test"],
    "textworld": ["test"],
    "mathpile": ["train"],
    "wikipedia18": ["train"],
    "wikipedia23": ["train"],
    "nq": ["train"],
}

def find_item(items, item):
    for i, x in enumerate(items):
        if x == item:
            return i
    return -1

def load_hotpotqa(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["hotpotqa"]

    dataset = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa", split=split)
    columns_to_remove = dataset.column_names
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": example["question"],
            "answer": example["golden_answers"],
            "data_source": "hotpotqa",
            "data_type": DataType.MULTIHOPQA,
            "idx": sample_id
        },
        with_indices=True
    )
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "idx", "data_source", "data_type"]
        ]
    )
    return dataset

def load_2wiki(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["2wiki"]

    dataset = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa", split=split)
    columns_to_remove = dataset.column_names
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": example["question"],
            "answer": example["golden_answers"],
            "data_source": "2wiki",
            "data_type": DataType.MULTIHOPQA,
            "idx": sample_id
        },
        with_indices=True
    )
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "idx", "data_source", "data_type"]
        ]
    )
    return dataset

def load_musique(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["musique"]

    dataset = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "musique", split=split)
    columns_to_remove = dataset.column_names
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": example["question"],
            "answer": example["golden_answers"],
            "data_source": "musique",
            "data_type": DataType.MULTIHOPQA,
            "idx": sample_id
        },
        with_indices=True
    )
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "idx", "data_source", "data_type"]
        ]
    )
    return dataset

def load_bamboogle(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["bamboogle"]

    dataset = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "bamboogle", split=split)
    columns_to_remove = dataset.column_names
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": example["question"],
            "answer": example["golden_answers"],
            "data_source": "bamboogle",
            "data_type": DataType.MULTIHOPQA,
            "idx": sample_id
        },
        with_indices=True
    )
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "idx", "data_source", "data_type"]
        ]
    )
    return dataset

def load_aime24(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["aime24"]
    dataset = datasets.load_dataset("HuggingFaceH4/aime_2024", split=split)
    columns_to_remove = dataset.column_names
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": example["problem"],
            "answer": [example["answer"]],
            "data_source": "aime24",
            "data_type": DataType.MATH,
            "idx": sample_id
        },
        with_indices=True
    )
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "idx", "data_source", "data_type"]
        ]
    )
    return dataset

def load_aime25(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["aime25"]
    dataset = datasets.load_dataset("math-ai/aime25", split=split)
    columns_to_remove = dataset.column_names
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": example["problem"],
            "answer": [example["answer"]],
            "data_source": "aime25",
            "data_type": DataType.MATH,
            "idx": sample_id
        },
        with_indices=True
    )
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "idx", "data_source", "data_type"]
        ]
    )
    return dataset

def load_math500(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["math500"]
    dataset = datasets.load_dataset("HuggingFaceH4/MATH-500", split=split)
    columns_to_remove = dataset.column_names
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": example["problem"],
            "answer": [example["answer"]],
            "data_source": "math500",
            "data_type": DataType.MATH,
            "idx": sample_id
        },
        with_indices=True
    )
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "idx", "data_source", "data_type"]
        ]
    )
    return dataset

def load_medqa(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["medqa"]

    dataset = datasets.load_dataset("openlifescienceai/medqa", split=split)
    columns_to_remove = dataset.column_names
    dataset = dataset.map(lambda example, sample_id:
        {
            "question": example["data"]["Question"],
            "answer": [example["data"]["Correct Option"]],
            "options": "\n".join([f'{k}): {v}' for k, v in example["data"]["Options"].items()]),
            "data_source": "medqa",
            "data_type": DataType.SINGLECHOICE,
            "idx": sample_id
        }, with_indices=True
    )
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "options", "idx", "data_source", "data_type"]
        ]
    )
    return dataset

def load_gpqa(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["gpqa"]

    dataset = datasets.load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split)
    columns_to_remove = dataset.column_names

    def map_fn(example, sample_id):
        import random
        random.seed(42)
        answer_idx = random.randint(0, 3)
        answer = [example["Correct Answer"]]
        candidate_answers = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        candidate_answers.insert(answer_idx, answer)
        index2letter = {0: "A", 1: "B", 2: "C", 3: "D"}

        return {
            "question": example["Question"],
            "answer": index2letter[answer_idx],
            "options": "\n".join([f'{index2letter[i]}): {v}' for i, v in enumerate(candidate_answers)]),
            "data_source": "gpqa",
            "data_type": DataType.SINGLECHOICE,
            "idx": sample_id
        }

    dataset = dataset.map(map_fn, with_indices=True)
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "options", "idx", "data_source", "data_type"]
        ]
    )
    return dataset

def load_webqsp(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["webqsp"]

    dataset = datasets.load_dataset("rmanluo/RoG-webqsp", split=split)
    columns_to_remove = dataset.column_names
    dataset = dataset.map(lambda example, sample_id:
        {
            "question": example["question"],
            "answer": example["answer"],
            "q_entity": example["q_entity"],
            "data_source": "webqsp",
            "data_type": DataType.KBQA,
            "idx": sample_id
        }, with_indices=True
    )
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "q_entity", "idx", "data_source", "data_type"]
        ]
    )
    return dataset

def load_wtq(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["wtq"]

    dataset = datasets.load_dataset("TableQAKit/WTQ", split=split)
    columns_to_remove = dataset.column_names
    dataset = dataset.map(lambda example, sample_id:
        {
            "question": example["question"],
            "answer": example["answer_text"],
            "table_id": example["table_id"],
            "data_source": "wtq",
            "data_type": DataType.TABLEQA,
            "idx": sample_id
        }, with_indices=True
    )
    dataset = dataset.remove_columns(
        [
            column for column in columns_to_remove 
            if column not in ["question", "answer", "table_id", "data_source", "data_type"]
        ]
    )
    return dataset

def load_textworld(split: str) -> datasets.Dataset:
    assert split in DATASET_INFOS["textworld"]

    path = Path(__file__).parents[2].joinpath("cache", "data", "textworld", "games.jsonl")
    examples = []
    with open(path, "r") as rfile:
        for line in rfile:
            example = json.loads(line)
            examples.append(example)
    dataset = []
    for example in examples:
        dataset.append({
            "question": example["question"],
            "answer": [],
            "data_source": "textworld",
            "data_type": DataType.TEXTGAME,
            "idx": example["idx"],
        })

    dataset = datasets.Dataset.from_list(dataset)
    return dataset

def load_mathpile(split: str) -> datasets.Dataset:
    assert split in ["train"]

    dataset = datasets.concatenate_datasets(
        [
            datasets.load_dataset("jnanliu/MathPile-Wikipedia", split=split), 
            datasets.load_dataset("jnanliu/MathPile-Textbooks", split=split),
        ],
        split=split
    )
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": None,
            "answer": None,
            "type": None,
            "level": None,
            "supporting_facts": None,
            "context": [
                [
                    {
                        "sample_id": sample_id, 
                        "doc_id": 0, 
                        "title": example["meta"].get("page_title", example["meta"]["book_name"]), 
                        "content": example["text"]
                    }
                ]
            ]
        },
        with_indices=True
    )
    dataset = dataset.remove_columns(["text", "subset", "meta", "file_path"])
    return dataset

def load_wikipedia18(split: str) -> datasets.Dataset:
    assert split in ["train"]

    dataset = datasets.load_dataset("jnanliu/wikipedia18", split=split)
    columns = dataset.column_names
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": None,
            "answer": None,
            "type": None,
            "level": None,
            "supporting_facts": None,
            "context": [
                [
                    {
                        "sample_id": example["contents"].split("\n")[0], 
                        "doc_id": 0, 
                        "title": example["contents"].split("\n")[0], 
                        "content": example["contents"].split("\n")[1]
                    }
                ]
            ]
        },
        with_indices=True
    )
    dataset.remove_columns(columns)
    return dataset

def load_wikipedia23(split: str) -> datasets.Dataset:
    assert split in ["train"]

    dataset = datasets.load_dataset("wikimedia/wikipedia", "20231101.en", split=split)
    columns = dataset.column_names
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": None,
            "answer": None,
            "type": None,
            "level": None,
            "supporting_facts": None,
            "context": [
                [
                    {
                        "sample_id": sample_id, 
                        "doc_id": 0, 
                        "title": example["title"], 
                        "content": example["text"]
                    }
                ]
            ]
        },
        with_indices=True
    )
    dataset.remove_columns(columns)
    return dataset

def load_nq(split: str) -> datasets.Dataset:
    assert split in ["train"]

    dataset = datasets.load_dataset("jnanliu/nq", split=split)
    columns = dataset.column_names
    dataset = dataset.map(lambda example, sample_id: 
        {
            "question": None,
            "answer": None,
            "type": None,
            "level": None,
            "supporting_facts": None,
            "context": [
                [
                    {
                        "sample_id": example["title"].split("\t")[1], 
                        "doc_id": 0, 
                        "title": example["title"].split("\t")[1], 
                        "content": " ".join(example["content"].split("\t")[1])
                    }
                ]
            ]
        },
        with_indices=True
    )
    dataset.remove_columns(columns)
    return dataset

def load_dataset(name: str, split: str = "train") -> datasets.Dataset:
    if name == "hotpotqa":
        dataset = load_hotpotqa(split)
    elif name == "2wiki":
        dataset = load_2wiki(split)
    elif name == "musique":
        dataset = load_musique(split)
    elif name == "bamboogle":
        dataset = load_bamboogle(split)
    elif name == "medqa":
        dataset = load_medqa(split)
    elif name == "aime24":
        dataset = load_aime24(split)
    elif name == "aime25":
        dataset = load_aime25(split)
    elif name == "math500":
        dataset = load_math500(split)
    elif name == "gpqa":
        dataset = load_gpqa(split)
    elif name == "webqsp":
        dataset = load_webqsp(split)
    elif name == "wtq":
        dataset = load_wtq(split)
    elif name == "textworld":
        dataset = load_textworld(split)
    elif name == "mathpile":
        dataset = load_mathpile(split)
    elif name == "wikipedia18":
        dataset = load_wikipedia18(split)
    elif name == "wikipedia23":
        dataset = load_wikipedia23(split)
    elif name == "nq":
        dataset = load_nq(split)
    else:
        raise ValueError("Unknown Dataset!")
    return dataset
