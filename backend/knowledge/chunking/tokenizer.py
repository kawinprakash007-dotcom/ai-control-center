class Tokenizer:

    """
    Tokenizer placeholder.

    Later this will use tiktoken
    or HuggingFace tokenizers.
    """

    def count_tokens(
        self,
        text: str
    ) -> int:

        return len(text.split())