import eflomal

from utils import Config, FileLoader, FileSaver, WordTokenizer, parse_args

class AlignProcessor:

    def preprocess_sentence(sentence: str, lang: str) -> str:
        """Return word tokens of sentence divided by space

        Args:
            sentence (str): Sentence to process
            lang (str): Sentence language

        Returns:
            str: Word tokens of sentence divided by space
        """
        tokenizer = WordTokenizer(lang)
        tokenized = " ".join(tokenizer.tokenize(sentence))

        return tokenized
    
    def preprocess_sentence_batch(batch: list[str], lang: str, filename: str) -> None:
        """Apply `preprocess_sentence` to list of sentences and save results

        Args:
            batch (list[str]): List of sentences to process
            lang (str): Language of sentences
            filename (str): File where to save results
        """
        
        tokenized = [AlignProcessor.preprocess_sentence(s, lang) for s in batch]
        FileSaver.save_sentences(tokenized, filename)

class Aligner:

    def align(
        src_filename: str,
        tgt_filename: str,
        fwd_filename: str,
        rev_filename: str
    ) -> None:
        """Apply eflomal aligner to specified files

        Args:
            src_filename (str): Name of file with word-tokenized sentences in source language
            tgt_filename (str): Name of file with word-tokenized sentences in target language
            fwd_filename (str): Name of forward alignment result file
            rev_filename (str): Name of reverse alignment result file
        """
        aligner = eflomal.Aligner()

        
        with open(src_filename, "r") as src_data, \
            open(tgt_filename, "r") as tgt_data:

            aligner.align(
                src_data,
                tgt_data,
                links_filename_fwd=fwd_filename,
                links_filename_rev=rev_filename)

if __name__=="__main__":

    config_path = parse_args()

    config = Config(config_path)

    src_sentences = FileLoader.load_sentences(config.translation_src)
    tgt_sentences = FileLoader.load_sentences(config.transliterate_fullpath)

    AlignProcessor.preprocess_sentence_batch(src_sentences, config.src_lang, config.align_preprocess_src_fullpath)
    AlignProcessor.preprocess_sentence_batch(tgt_sentences, config.tgt_lang, config.align_preprocess_tgt_fullpath)

    Aligner.align(config.align_preprocess_src_fullpath, config.align_preprocess_tgt_fullpath, config.align_fwd_fullpath, config.align_rev_fullpath)
