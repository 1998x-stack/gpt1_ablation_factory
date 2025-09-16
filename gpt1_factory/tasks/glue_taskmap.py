GLUE_TASKS = {
    "sst2": {"num_labels": 2, "text_cols": ("sentence", None), "metric": "acc"},
    "mnli": {"num_labels": 3, "text_cols": ("premise", "hypothesis"), "metric": "acc"},
    "mrpc": {"num_labels": 2, "text_cols": ("sentence1", "sentence2"), "metric": "f1"},
    "qqp":  {"num_labels": 2, "text_cols": ("question1", "question2"), "metric": "f1"},
    "cola": {"num_labels": 2, "text_cols": ("sentence", None), "metric": "matthews"},
    "stsb": {"num_labels": 1, "text_cols": ("sentence1", "sentence2"), "metric": "pearson"},
}
