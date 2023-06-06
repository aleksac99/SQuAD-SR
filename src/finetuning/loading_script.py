"""Loading script adapted from https://huggingface.co/datasets/squad/blob/main/squad.py."""

import json
import os

import datasets

_DESCRIPTION = """Serbian version of SQuAD v1.1 dataset."""

class SquadSr(datasets.GeneratorBasedBuilder):
    """SQuAD Dataset automatically translated to Serbian language"""

    VERSION = datasets.Version("1.1.0")

    def _info(self):
        features = datasets.Features(
            {
                "id": datasets.Value("string"),
                "title": datasets.Value("string"),
                "context": datasets.Value("string"),
                "question": datasets.Value("string"),
                "answers": datasets.Sequence({
                    "text": datasets.Value("string"),
                    "answer_start": datasets.Value("int32")
                })
            }
        )
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            supervised_keys=None
        )

    def _split_generators(self, dl_manager):
        urls = {}
        if self.config.data_files.get("train","") != "":
            urls["train"] = self.config.data_files["train"]
        if self.config.data_files.get("dev","") != "":
            urls["dev"] = self.config.data_files["dev"]

        data_dir = dl_manager.extract(urls)
        generators = []
        if "train" in data_dir:
            generators.append(
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                
                gen_kwargs={"filepath": data_dir["train"][0]},))
        if "dev" in data_dir:
            generators.append(
                datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"filepath": data_dir["dev"][0]}))

        return generators

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8") as f:
            squad_sr = json.load(f)
            for article in squad_sr["data"]:
                title = article["title"].strip()
                for paragraph in article["paragraphs"]:
                    context = paragraph["context"].strip()
                    for qa in paragraph["qas"]:
                        question = qa["question"].strip()
                        id_ = qa["id"]

                        answer_starts = [a["answer_start"]
                                         for a in qa["answers"]]
                        answer_texts = [a["text"] for a in qa["answers"]]

                        yield id_, {
                            "title": title,
                            "context": context,
                            "question": question,
                            "id": id_,
                            "answers": {"answer_start": answer_starts,
                                        "text": answer_texts}
                        }