from tqdm import tqdm

from utils import (Config, FileLoader, FileSaver, SentenceProcessor,
                   SentenceTokenizer, WordTokenizer, parse_args)


class RetrieveProcessor:

    def retrieve_preprocess(
            src_sentences: list[str],
            tgt_sentences: list[str],
            alignments: list[str]) -> dict:
        """Transform data into format convenient for retrieval step

        Args:
            src_sentences (list[str]): List of sentences in source language
            tgt_sentences (list[str]): List of sentences in target language
            alignments (list[str]): List with alignments of source and target sentences

        Returns:
            dict: Data in format convenient for retrieval
        """

        res = {s:
               {"translation": t, "alignment": a}
               for s, t, a in zip(src_sentences, tgt_sentences, alignments)}

        return res


class SQuADBuilder:

    def get_tgt_context_alignment(
        src_context: str,
        src2tgt_map: dict,
        src_lang: str,
        tgt_lang: str
        ) -> tuple[str, dict, list]:
        """Calculate context alignment from given sentence alignments

        Args:
            src_context (str): Context in source language
            src2tgt_map (dict): Previous steps results
            src_lang (str): Source language
            tgt_lang (str): Target language

        Returns:
            tuple[str, dict, list]: Target context, context alignment and list of target sentences
        """

        src_sent_tokenizer = SentenceTokenizer(src_lang)
        tgt_sent_tokenizer = SentenceTokenizer(tgt_lang)

        src_sentences = src_sent_tokenizer.tokenize(src_context)
        tgt_sentences = [src2tgt_map[SentenceProcessor.process_sentence(s)]['translation'] for s in src_sentences]
        alignments = [src2tgt_map[SentenceProcessor.process_sentence(s)]['alignment'] for s in src_sentences]

        context_alignment = SQuADBuilder.compute_context_alignment(src_sentences, tgt_sentences, alignments, src_lang, tgt_lang)

        tgt_context = tgt_sent_tokenizer.detokenize(tgt_sentences)

        return (tgt_context, context_alignment, tgt_sentences)

    def compute_context_alignment(
        src_sentences: list[str],
        tgt_sentences: list[str],
        alignments: list[str],
        src_lang: str,
        tgt_lang: str,
        ) -> dict:
        """Context alignment calculation algorithm

        Args:
            src_sentences (list[str]): List of sentences in source language
            tgt_sentences (list[str]): List of sentences in target language
            alignments (list[str]): List of alignments
            src_lang (str): Source language
            tgt_lang (str): Target language

        Returns:
            dict: Source to target word alignment map
        """
        
        src_word_tokenizer = WordTokenizer(src_lang)
        tgt_word_tokenizer = WordTokenizer(tgt_lang)
        src_shift, tgt_shift = 0, 0
        context_alignments = {}

        for s, t, a in zip(src_sentences, tgt_sentences, alignments):

            for idxs in a.split():
                src_idx = int(idxs.split('-')[0]) + src_shift
                tgt_idx = int(idxs.split('-')[1]) + tgt_shift

                if context_alignments.get(src_idx) is not None:
                    context_alignments[src_idx].append(tgt_idx)
                else:
                    context_alignments[src_idx] = [tgt_idx]
                
            src_shift += len(src_word_tokenizer.tokenize(s))
            tgt_shift += len(tgt_word_tokenizer.tokenize(t))

        return context_alignments

    def get_mappings(context: str, lang: str) -> tuple[dict, dict]:
        """Calculate character-to-word and word-to-character index mappings.

        Args:
            context (str): Context
            lang (str): Language

        Returns:
            tuple[dict, dict]: word-to-character and character-to-word index mappings
        """
        tokenizer = WordTokenizer(lang)

        word_tokens = tokenizer.tokenize(context)
        
        tok2char = dict()
        char2tok = dict()

        start = 0
        for tok_idx, word in enumerate(word_tokens):
            if word in ["``", "''"]: # There were some problems with these characters
                word = "\""
            char_idx = context.find(word, start)
            if char_idx == -1:
                print(word)
                continue # Skip the sample
                char_idx = start + 1 # Least evil

            tok2char[tok_idx] = char_idx
            char2tok[char_idx] = tok_idx
            start = max(0, char_idx + len(word) - 1)

        return tok2char, char2tok

    def build_squad(squad_file:str, squad_tgt_file: str, src2tgt_map: dict, src_lang: str, tgt_lang: str) -> dict:
        """Build SQuAD file in target language.

        Args:
            squad_file (str): Path to SQuAD file
            squad_tgt_file (str): Name of SQuAD file in target language
            src2tgt_map (dict): Dict that maps source sentence to target sentence and alignments
            src_lang (str): Source language
            tgt_lang (str): Target language

        Returns:
            dict: Dict in form SQuAD in target language
        """
        skipped = 0
        dataset = FileLoader.load_squad(squad_file)
        translated_dataset = {
            'version': dataset['version'],
            'data': []
        }
        tgt_word_tokenizer = WordTokenizer(tgt_lang)
        for data in tqdm(dataset["data"]):

            translated_data = {
                "title": src2tgt_map[data["title"]]["translation"],
                "paragraphs": []
            }

            for paragraph in data["paragraphs"]:
                
                
                translated_context, alignments, tgt_sentences = SQuADBuilder.get_tgt_context_alignment(paragraph['context'], src2tgt_map, src_lang, tgt_lang)

                translated_paragraph = {
                    "context": translated_context,
                    "qas": []
                }

                _, src_char2tok = SQuADBuilder.get_mappings(paragraph["context"], src_lang)
                tgt_tok2char, _ = SQuADBuilder.get_mappings(translated_context, tgt_lang)

                for qa in paragraph["qas"]:

                    translated_qa = {
                        "question": src2tgt_map[SentenceProcessor.process_sentence(qa["question"])]["translation"],
                        "id": qa["id"],
                        "answers": []
                    }

                    for answer in qa["answers"]:

                        answer_end = answer["answer_start"] + len(answer["text"])

                        tmp = 0
                        while True:
                            src_tok_answer_start = src_char2tok.get(answer["answer_start"] - tmp)
                            if src_tok_answer_start is not None:
                                break
                            tmp += 1

                        while True:
                            src_tok_answer_end = src_char2tok.get(answer_end)
                            if src_tok_answer_end is not None:
                                break
                            elif answer_end > len(paragraph["context"]):
                                src_tok_answer_end = max([v for _, v in src_char2tok.items()])
                                break
                            answer_end += 1

                        tgt_tokens = []
                        for tok_idx in range(src_tok_answer_start, src_tok_answer_end):

                            # Find alignments
                            tgt_tokens += alignments.get(tok_idx, [])

                        if tgt_tokens == []:
                            try:
                                tgt_tokens = alignments[src_tok_answer_end]
                            except:
                                skipped += 1
                                continue

                        tgt_tokens_words = []
                        for xx in tgt_sentences:
                            tgt_tokens_words += tgt_word_tokenizer.tokenize(xx)

                        min_aligned, max_aligned = min(tgt_tokens), max(tgt_tokens) + 1

                        try:
                            tgt_tok_answer_start, tgt_tok_answer_end = tgt_tok2char[min_aligned], tgt_tok2char[max_aligned]
                        except:
                            skipped +=1
                            continue

                        translated_text = translated_context[tgt_tok_answer_start:tgt_tok_answer_end].strip()

                        translated_qa["answers"].append({
                            "answer_start": tgt_tok_answer_start,
                            "text": translated_text
                        })
                    
                    if translated_qa["answers"] != []:
                        translated_paragraph["qas"].append(translated_qa)

                translated_data["paragraphs"].append(translated_paragraph)
            
            translated_dataset["data"].append(translated_data)

            
        FileSaver.save_squad(translated_dataset, squad_tgt_file)

        return translated_dataset

if __name__ == '__main__':

    config_path = parse_args()

    config = Config(config_path)

    src_sentences = FileLoader.load_sentences(config.translation_src)
    tgt_sentences = FileLoader.load_sentences(config.transliterate_fullpath)
    alignments = FileLoader.load_alignments(config.align_sym_fullpath)

    src2tgt_map = RetrieveProcessor.retrieve_preprocess(src_sentences, tgt_sentences, alignments)
    squad_sr = SQuADBuilder.build_squad(config.squad, config.retrieve_fullpath, src2tgt_map, config.src_lang, config.tgt_lang)
