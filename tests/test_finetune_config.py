from gpt1_factory.cli.finetune import load_cfg_with_overrides


def test_data_override_preserves_included_tokenizer_artifact():
    cfg = load_cfg_with_overrides(
        "configs/finetune_glue.yaml",
        ["data.task=sst2"],
    )

    assert cfg["data"]["name"] == "glue"
    assert cfg["data"]["task"] == "sst2"
    assert (
        cfg["data"]["tokenizer_artifact"]
        == "runs/exp_pretrain_books/tokenizer"
    )
