from cyrtranslit import to_latin

from utils import Config, FileLoader, FileSaver, parse_args


class Transliterator:
    
    def transliterate(text: str) -> str:
        """Perform cyrillic to latin transliteration

        Args:
            text (str): Text to transliterate

        Returns:
            str: Transliterated text
        """
        return to_latin(text)

    def transliterate_batch(batch: list[str]) -> list[str]:
        """Perform batch transliteration

        Args:
            batch (list[str]): List of texts to transliterate

        Returns:
            list[str]: List of transliterated texts
        """
        return [Transliterator.transliterate(b) for b in batch]



if __name__=='__main__':
    
    config_path = parse_args()

    config = Config(config_path)
    sentences = FileLoader.load_sentences(config.translation_tgt)

    transliterated_sentences = Transliterator.transliterate_batch(sentences)

    FileSaver.save_sentences(transliterated_sentences, config.transliterate_fullpath)