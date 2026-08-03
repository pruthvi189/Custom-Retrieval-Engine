from engine import chunk_text, graph_embedding


class TestChunkText:
    def test_short_text_unchanged(self):
        text = "one two three four"
        assert chunk_text(text) == [text]

    def test_empty_and_whitespace(self):
        assert chunk_text("") == []
        assert chunk_text("   \n\t  ") == []

    def test_step_is_chunk_minus_overlap(self):
        words = [str(i) for i in range(250 * 3)]
        text = " ".join(words)
        chunks = chunk_text(text, 250, 30)
        assert len(chunks) > 1
        first = chunks[0].split()
        second = chunks[1].split()
        assert len(first) == 250
        assert len(second) == 250
        # second chunk starts at word index 220, so its first 30 words overlap
        assert second[:30] == first[220:250]

    def test_reassembles_without_losing_words(self):
        text = " ".join("w%d" % i for i in range(700))
        chunks = chunk_text(text, 250, 30)
        joined = " ".join(chunks)
        for i in range(700):
            assert ("w%d" % i) in joined

    def test_exact_word_boundaries(self):
        text = "alpha beta gamma delta epsilon"
        assert chunk_text(text, 3, 0) == ["alpha beta gamma", "delta epsilon"]

    def test_single_chunk_when_barely_under(self):
        words = ["x"] * 250
        assert chunk_text(" ".join(words)) == [" ".join(words)]


class TestGraphEmbedding:
    def test_deterministic(self):
        assert graph_embedding("binary search tree algorithm") == graph_embedding(
            "binary search tree algorithm"
        )

    def test_cs_dominant(self):
        emb = graph_embedding("binary search tree data structure")
        assert emb[0] > emb[4] and emb[0] > emb[8] and emb[0] > emb[12]

    def test_degenerate_text_baseline(self):
        emb = graph_embedding("")
        assert emb == [0.08] * 16

    def test_bounded_values(self):
        emb = graph_embedding("food pizza pasta restaurant cheese tomato")
        assert all(0.05 <= v <= 0.95 for v in emb)

    def test_category_regions(self):
        for text, idx in [
            ("data structures algorithms", 0),
            ("calculus integral derivative", 4),
            ("pizza recipe restaurant", 8),
            ("basketball team competition", 12),
        ]:
            emb = graph_embedding(text)
            assert emb[idx] > 0.4
