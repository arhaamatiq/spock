from langchain_core.documents import Document


def test_profile_question_detection_handles_vague_recruiter_questions():
    from spock_rag.retrieval import is_candidate_profile_question

    assert is_candidate_profile_question("Where are you from?")
    assert is_candidate_profile_question("Where is Arhaam from?")
    assert is_candidate_profile_question("What has AgentGate shipped?")


def test_profile_questions_get_arhaam_specific_query_expansion():
    from spock_rag.retrieval import build_retrieval_queries

    queries = build_retrieval_queries("Where are you from?")

    assert queries[0] == "Where are you from?"
    assert any("Arhaam Atiq" in query for query in queries[1:])
    assert any("Bangalore India" in query for query in queries[1:])


def test_profile_context_documents_load_from_configured_docs_dir(tmp_path):
    from spock_rag.retrieval import load_profile_context_documents

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "identity_snapshot.txt").write_text(
        "My name is Arhaam Atiq. I go to San Jose State University.",
        encoding="utf-8",
    )
    (docs_dir / "origin_story.txt").write_text(
        "I am an international student from Bangalore, India.",
        encoding="utf-8",
    )

    docs = load_profile_context_documents(docs_dir)

    assert len(docs) == 2
    assert any("Bangalore, India" in doc.page_content for doc in docs)


def test_profile_aware_retrieval_uses_fallback_when_vector_search_fails(
    monkeypatch,
    tmp_path,
):
    from spock_rag.config import reset_settings
    from spock_rag import retrieval

    docs_dir = tmp_path / "docs"
    docs_dir.mkdir()
    (docs_dir / "origin_story.txt").write_text(
        "I am an international student from Bangalore, India.",
        encoding="utf-8",
    )
    monkeypatch.setenv("DOCS_DIR", str(docs_dir))
    reset_settings()
    retrieval._load_profile_documents_from_dir.cache_clear()

    def fail_vector_search(*args, **kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(retrieval, "retrieve_with_scores", fail_vector_search)

    results = retrieval.retrieve_profile_aware_documents("where are you from?", k=4)

    assert len(results) == 1
    doc, score = results[0]
    assert isinstance(doc, Document)
    assert score == 1.0
    assert "Bangalore, India" in doc.page_content
