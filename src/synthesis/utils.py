import argparse
import json
import nltk
import os

class FileSaver:

    def save_sentences(sentences: list[str], filename: str) -> None:
        """Write list of sentences to file, separated by newline character

        Args:
            sentences (list[str]): List of sentences
            filename (str): Name of file where to write
        """
        text = '\n'.join(sentences)
        with open(filename, 'w') as f:
            f.write(text)

    def save_squad(data: dict, filename: str) -> None:
        """Write squad-like dict to JSON file

        Args:
            data (dict): SQuAD-like file
            filename (str): Name of file where to write
        """

        with open(filename, 'w') as f:
            json.dump(data, f, ensure_ascii=False)

class FileLoader:
    
    def load_sentences(filename: str) -> list[str]:
        """Load sentences from text file into a list

        Args:
            filename (str): File name

        Returns:
            list[str]: List of loaded sentences
        """

        with open(filename, 'r') as f:
            text = f.read()

        sentences = text.split('\n')

        return sentences

    def load_alignments(filename: str) -> list[str]:
        """Load word alignments
        Function internally calls `load_sentence`

        Args:
            filename (str): Name of alignments file

        Returns:
            list[str]: List of alignments
        """

        return FileLoader.load_sentences(filename)

    def load_squad(filename: str) -> dict:
        """Load SQuAD dataset from file

        Args:
            filename (str): Path to SQuAD file

        Returns:
            dict: SQuAD dataset
        """
        with open(filename, 'r') as f:
            squad = json.load(f)

        return squad


class SentenceTokenizer:

    def __init__(self, lang: str) -> None:

        self.lang = lang
        self.default_lang = 'english'
    
    def tokenize(self, text: str) -> list[str]:
        """Split text into sentences

        Args:
            text (str): Text to split into sentences

        Returns:
            list[str]: List of sentences
        """
        try:
            tokens = nltk.sent_tokenize(text, language=self.lang)
        except:
            tokens = nltk.sent_tokenize(text, language=self.default_lang)

        return tokens

    def detokenize(self, sentences: list[str]) -> str:
        """Concatenate list of sentences into text

        Args:
            sentences (list[str]): List of sentences

        Returns:
            str: Concatenated sentences
        """
        return " ".join(sentences)

    def batch_tokenize(self, batch: list[str]) -> list:
        """Sentence-tokenize a batch of texts

        Args:
            batch (list[str]): Batch of texts for sentence-tokenization

        Returns:
            list: List containing lists of sentence-tokenized text
        """
        return [self.tokenize(text) for text in batch]


class WordTokenizer:
    
    def __init__(self, lang: str) -> None:

        self.lang = lang
        self.default_lang = 'english'

    def tokenize(self, sentence: str) -> list[str]:
        """Tokenize sentence into words

        Args:
            sentence (str): Sentence to tokenize

        Returns:
            list[str]: List of words
        """
        try:
            tokens = nltk.word_tokenize(sentence, language=self.lang)
        except:
            tokens = nltk.word_tokenize(sentence, language=self.default_lang)

        return tokens

    def batch_tokenize(self, batch: list[str]) -> list:
        """Perform word tokenization on list of sentences

        Args:
            batch (list[str]): List of sentenes to tokenize

        Returns:
            list: List of tokenized sentences
        """

        return [self.tokenize(sentence) for sentence in batch]

class Config:
    
    def __init__(self, file: str) -> None:

        self.tmp_folder = "tmp"
        
        with open(file, 'r') as f:
            config = json.load(f)

        self.overwrite = config.get('overwrite')

        self.squad = config.get('squad_file')

        # Folders
        self.data_folder = config.get('data_folder')
        self.translate_folder = config.get('translate_folder')
        self.align_folder = config.get('align_folder')
        self.retrieve_folder = config.get('retrieve_folder')

        # Translate
        self.translate_src_filename = config.get('translate_src_filename')
        self.translate_tgt_filename = config.get('translate_tgt_filename')
        self.translate_model = config.get('translate_model')
        self.src_lang = config.get('translate_src_lang')
        self.tgt_lang = config.get('translate_tgt_lang')
        self.start_idx = config.get('translate_start_idx')
        self.end_idx = config.get('translate_end_idx')

        # Transliterate
        self.transliterate = config.get('transliterate')
        self.transliterate_filename = config.get('transliterate_filename')

        # Align
        self.align_tok_src = config.get('align_tok_src')
        self.align_tok_tgt = config.get('align_tok_tgt')
        self.align_fwd_filename = config.get('align_fwd_file')
        self.align_rev_filename = config.get('align_rev_file')
        self.align_sym_filename = config.get('align_sym_file')

        # Retrieve
        self.retrieve_filename = config.get('retrieve_filename')

        self._create_file_structure()

    def _create_file_structure(self) -> None:
        """Create folders required for further files saving

        Raises:
            FileExistsError: Folder already exists
        """

        try:
            os.makedirs(self.data_folder, exist_ok=self.overwrite)
            os.makedirs(os.path.join(self.data_folder, self.translate_folder), exist_ok=self.overwrite)
            os.makedirs(os.path.join(self.data_folder, self.align_folder), exist_ok=self.overwrite)
            os.makedirs(os.path.join(self.data_folder, self.retrieve_folder), exist_ok=self.overwrite)
        except:
            raise FileExistsError()

    @property
    def translation_src(self):
        return os.path.join(self.data_folder, self.translate_folder, self.translate_src_filename)
        
    @property
    def translation_tgt(self):
        return os.path.join(self.data_folder, self.translate_folder, self.translate_tgt_filename)

    @property
    def transliterate_fullpath(self):
        return os.path.join(self.data_folder, self.translate_folder, self.transliterate_filename) if self.transliterate else os.path.join(self.data_folder, self.translate_folder, self.translate_tgt_filename)

    @property
    def align_preprocess_src_fullpath(self):
        return os.path.join(self.data_folder, self.align_folder, self.align_tok_src)
        
    @property
    def align_preprocess_tgt_fullpath(self):
        return os.path.join(self.data_folder, self.align_folder, self.align_tok_tgt)

    @property
    def align_fwd_fullpath(self):
        return os.path.join(self.data_folder, self.align_folder, self.align_fwd_filename)
        
    @property
    def align_rev_fullpath(self):
        return os.path.join(self.data_folder, self.align_folder, self.align_rev_filename)
        
    @property
    def align_sym_fullpath(self):
        return os.path.join(self.data_folder, self.align_folder, self.align_sym_filename)
        
    @property
    def retrieve_fullpath(self):
        return os.path.join(self.data_folder, self.retrieve_folder, self.retrieve_filename)

class SQuADProcessor:

    def extract_contexts(squad: dict) -> list[str]:
        """Extract texts labeled as `context` in SQuAD Dataset

        Args:
            squad (dict): SQuAD Dataset

        Returns:
            list[str]: List of contexts
        """
        contexts = [p['context'] for d in squad['data'] for p in d['paragraphs']]
        return contexts

    def extract_questions(squad: dict) -> list[str]:
        """Extract texts labeled as `question` in SQuAD Dataset

        Args:
            squad (dict): SQuAD Dataset

        Returns:
            list[str]: List of contexts
        """
        questions = [qa['question'] for d in squad['data'] for p in d['paragraphs'] for qa in p['qas']]
        return questions
        
    def extract_answers(squad: dict) -> list[str]:
        """Extract texts labeled as `answer` in SQuAD Dataset

        Args:
            squad (dict): SQuAD Dataset

        Returns:
            list[str]: List of contexts
        """
        answers = [a['text'] for d in squad['data'] for p in d['paragraphs'] for qa in p['qas'] for a in qa['answers']]
        return answers

class SentenceProcessor:

    def process_sentence(sentence: str) -> str:
        """Remove new lines and strip sentence

        Args:
            sentence (str): Sentence to process

        Returns:
            str: Processed sentence
        """
        
        # Remove newline
        sentence = sentence.replace("\n", " ")
        sentence = sentence.replace("\r", " ")

        # Strip
        sentence = sentence.strip()

        return sentence

        
    def process_sentence_batch(sentences: list[str]) -> list[str]:
        """Apply `process_sentence` to batch of sentences

        Args:
            sentences (list[str]): List of sentences

        Returns:
            list[str]: List of processed sentences
        """

        return [SentenceProcessor.process_sentence(s) for s in sentences]

def parse_args() -> str:
    """Parse arguments

    Returns:
        str: Config path
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path')
    args = parser.parse_args()
    return args.config_path