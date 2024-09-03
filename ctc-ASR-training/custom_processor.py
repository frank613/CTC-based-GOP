from transformers import Wav2Vec2CTCTokenizer, Wav2Vec2Processor, Wav2Vec2FeatureExtractor
import warnings

class My_Wav2Vec2CTCTokenizer(Wav2Vec2CTCTokenizer):
 
    def __init__(
        self,
        vocab_file,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        word_delimiter_token="|",
        replace_word_delimiter_char=" ",
        do_lower_case=False,
        **kwargs
    ):
        super().__init__(
            vocab_file,
            bos_token="<s>",
            eos_token="</s>",
            unk_token="<unk>",
            pad_token="<pad>",
            word_delimiter_token="|",
            replace_word_delimiter_char=" ",
            do_lower_case=False,
            **kwargs
        )

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (string), without using the tokenizer. As the input are already list of tokens not words.
        """
        if self.do_lower_case:
            text = text.upper()

        return list(text.replace(" ", ""))
    

class My_Wav2Vec2Processor(Wav2Vec2Processor):
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):

        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(pretrained_model_name_or_path, **kwargs)
        tokenizer = My_Wav2Vec2CTCTokenizer.from_pretrained(pretrained_model_name_or_path, **kwargs)

        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)