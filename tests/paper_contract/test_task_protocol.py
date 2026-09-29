from gpt1_factory.data.text_bpe import BPEBuilder
from gpt1_factory.tasks.protocol import GPT1TaskProtocol


def _protocol(tmp_path, max_len=16):
    builder = BPEBuilder(tmp_path / "bpe", vocab_size=200, min_freq=1)
    tok = builder.train(
        [
            "alpha beta gamma delta",
            "one two three four",
            "answer choice",
        ]
    )
    return tok, GPT1TaskProtocol(tok, max_len=max_len)


def test_classification_uses_start_and_classify_tokens(tmp_path):
    tok, protocol = _protocol(tmp_path)
    encoding = protocol.encode_classification("alpha beta")

    assert encoding.input_ids[0] == tok.token_to_id("_start_")
    assert encoding.input_ids[-1] == tok.token_to_id("_classify_")
    assert encoding.classify_position == len(encoding.input_ids) - 1


def test_entailment_inserts_delimiter(tmp_path):
    tok, protocol = _protocol(tmp_path)
    encoding = protocol.encode_entailment(
        "alpha beta",
        "one two",
    )

    assert tok.token_to_id("_delimiter_") in encoding.input_ids
    assert encoding.input_ids[-1] == tok.token_to_id("_classify_")


def test_similarity_emits_both_orders(tmp_path):
    _, protocol = _protocol(tmp_path)
    forward, reverse = protocol.encode_similarity(
        "alpha beta",
        "one two",
    )

    assert forward.input_ids != reverse.input_ids
    assert forward.classify_position == len(forward.input_ids) - 1
    assert reverse.classify_position == len(reverse.input_ids) - 1


def test_truncation_never_drops_classify_token(tmp_path):
    tok, protocol = _protocol(tmp_path, max_len=5)
    encoding = protocol.encode_entailment(
        "alpha beta gamma delta alpha beta",
        "one two three four one two",
    )

    assert len(encoding.input_ids) <= 5
    assert encoding.input_ids[-1] == tok.token_to_id("_classify_")
    assert encoding.classify_position == len(encoding.input_ids) - 1
