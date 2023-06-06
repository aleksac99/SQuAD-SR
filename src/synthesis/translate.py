import torch
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from utils import (Config, FileLoader, FileSaver, SentenceProcessor,
                   SentenceTokenizer, SQuADProcessor, parse_args)


class Translator:

    LANGUAGE2NLLB = {
        "english": "eng_Latin",
        "serbian": "srp_Cyrl"
    }

    def __init__(self,
                 src_lang: str,
                 tgt_lang: str,
                 model_name: str,
                 start_idx: int,
                 end_idx: int
                 ) -> None:

        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.model_name = model_name
        self.start_idx = start_idx
        self.end_idx = end_idx

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self._load_model()

    def _load_model(self) -> None:
        """Load model from HuggingFace

        Returns:
            transformers.pipeline: HuggingFace translation pipeline
        """
        
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.translation_pipeline = pipeline("translation",
                                        model=self.model,
                                        tokenizer=self.tokenizer,
                                        src_lang=self.LANGUAGE2NLLB[self.src_lang],
                                        tgt_lang=self.LANGUAGE2NLLB[self.tgt_lang],
                                        device=self.device
                                        )

        print(f"Device: {self.translation_pipeline.device}")
        print(f"Model: {self.model_name}")

    def translate(
        self,
        src_sentences: list[str],
        filename: str,
        save_every: int = 1000
        ) -> list[str]:
        """Wrapper around HuggingFace translation pipeline

        Args:
            src_sentences (list[str]): List of sentences in source
            filename (str): Name of file where sentences are stored
            save_every (int, optional): Number of sentences after which the saving is performed. Defaults to 1000.

        Returns:
            list[str]: List of translated sentences
        """

        start_idx = 0 if self.start_idx is None else self.start_idx
        end_idx = len(src_sentences) if self.end_idx is None else self.end_idx
        tgt_sentences = []
        
        for i, s in tqdm(enumerate(src_sentences[start_idx:end_idx]), total=len(src_sentences[start_idx:end_idx])):
            tgt_sentences.append(self.translation_pipeline(s)[0]["translation_text"])
            if (i+1) % save_every == 0:
                FileSaver.save_sentences(src_sentences, filename)

        FileSaver.save_sentences(src_sentences, filename)
        print(f"Translated {len(tgt_sentences)} sentences.")

        return tgt_sentences


if __name__=='__main__':

    config_path = parse_args()

    config = Config(config_path)
    squad = FileLoader.load_squad(config.squad)
    tokenizer = SentenceTokenizer('english')

    # Extract data from SQuAD
    contexts = SQuADProcessor.extract_contexts(squad)
    questions = SQuADProcessor.extract_questions(squad)
    answers = SQuADProcessor.extract_answers(squad)

    # Extract sentences
    src_sentences = [s for c in tokenizer.batch_tokenize(contexts) for s in c]
    src_sentences += questions
    src_sentences += answers

    # Preprocess sentences
    src_sentences = SentenceProcessor.process_sentence_batch(src_sentences)
    
    # Remove duplicates
    src_sentences = set(src_sentences)
    src_sentences = list(src_sentences)

    FileSaver.save_sentences(src_sentences, config.translation_src)
    
    translator = Translator(config.src_lang, config.tgt_lang, config.translate_model, config.start_idx, config.end_idx)

    tgt_sentences = translator.translate(src_sentences, config.translation_tgt)

    tgt_sentences = SentenceProcessor.process_sentence_batch(tgt_sentences)

    FileSaver.save_sentences(tgt_sentences, config.translation_tgt)