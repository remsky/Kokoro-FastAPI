def get_vocab():
    """Get the vocabulary dictionary mapping characters to token IDs"""
    _pad = "$"
    _punctuation = ';:,.!?¡¿—…"«»"" '
    _letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
    _letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"

    # Create vocabulary dictionary
    symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
    return {symbol: i for i, symbol in enumerate(symbols)}


# Initialize vocabulary
VOCAB = get_vocab()


def tokenize(phonemes: str) -> list[int]:
    """Convert phonemes string to token IDs

    Args:
        phonemes: String of phonemes to tokenize

    Returns:
        List of token IDs
    """
    # Strip phonemes to remove leading/trailing spaces that could cause artifacts
    phonemes = phonemes.strip()
    return [i for i in map(VOCAB.get, phonemes) if i is not None]
