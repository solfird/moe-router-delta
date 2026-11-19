import overlap_audit as oa


def test_shingles_are_normalised_word_windows():
    assert oa.shingles("The cat, the DOG.", 2) == {("the", "cat"), ("cat", "the"), ("the", "dog")}
    assert oa.shingles("one two", 3) == set()


def test_hits_counts_test_texts_sharing_a_window_with_training():
    train = oa.index(["alpha beta gamma delta epsilon"], 3)
    tests = ["x beta gamma delta y", "nothing in common here at all", "alpha beta"]
    assert oa.hits(tests, train, 3) == [0]


def test_body_drops_chat_template_and_instruction():
    chat = (
        "<|im_start|>user\nClassify this item into one of: A, B. Reply with the label only."
        "\n\nThe real text.<|im_end|>\n<|im_start|>assistant\n"
    )
    assert oa.body(chat) == "The real text."
    assert oa.body("Summarize this.\n\nArticle body") == "Article body"
