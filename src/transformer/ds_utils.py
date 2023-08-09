from datasets import load_dataset


def load_opus_en_id():
    opus = load_dataset("opus100", "en-id")
    # remove outer "translation key"
    opus = opus.map(lambda row: row["translation"], remove_columns="translation")
    return opus
