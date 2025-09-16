from gpt1_factory.data.text_bpe import BPEBuilder

def test_bpe_roundtrip(tmp_path):
    data = ["hello world", "another sample"]
    builder = BPEBuilder(tmp_path / "bpe", 2000, 2)
    tok = builder.train(data)
    t = tok.encode("hello world").ids
    assert len(t) > 0
