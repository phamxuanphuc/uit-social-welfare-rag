from p01kit.data import (ANSWER_KEYS, HOLDOUT_POLICIES, N_CONTEXT, SPLIT_FILES, applicable, build_dataset,
                         dataset_stats, gold_answer, read_jsonl, resolve)


def test_corpus_integrity(corpus):
    codes = [p.doc_code for p in corpus.policies]
    assert len(codes) == len(set(codes)), "mã văn bản phải duy nhất"
    for p in corpus.policies:
        assert p.region in corpus.regions and p.topic in corpus.topics
        assert p.status in ("active", "expired")
        assert (p.expiry_date is None) == p.is_active
        assert len(p.who) >= 3 and len(p.situation) >= 3 and p.required_documents


def test_resolve_prefers_active_then_local(corpus):
    dn_old, dn_new = corpus.by_id("DN-TAINAN-2023"), corpus.by_id("DN-TAINAN-2026")
    assert resolve("tai_nan", "DN", [dn_old, dn_new]) is dn_new
    assert resolve("tai_nan", "DN", [dn_old]) is dn_old
    hcm_tro, tq_tro = corpus.by_id("HCM-TRO-2024"), corpus.by_id("TQ-TRO-2022")
    assert resolve("tien_tro", "HCM", [tq_tro, hcm_tro]) is hcm_tro
    assert resolve("tien_tro", "HCM", [corpus.by_id("DN-TRO")]) is None
    assert resolve(None, "HCM", corpus.policies) is None


def test_build_is_deterministic(corpus):
    assert build_dataset(corpus) == build_dataset(corpus)


def test_committed_files_match_builder(splits):
    for name, rows in splits.items():
        assert read_jsonl(SPLIT_FILES[name]) == rows, f"Chạy lại `python -m p01kit.data` để cập nhật {name}"


def test_no_group_leakage(splits):
    groups = {k: {r["group_id"] for r in v} for k, v in splits.items()}
    assert not groups["train"] & groups["test"]
    assert not groups["train"] & groups["val"]
    assert not groups["val"] & groups["test"]


def test_holdout_policies_never_seen_in_training(splits, corpus):
    hold = set(HOLDOUT_POLICIES)
    for split in ("train", "val"):
        for r in splits[split]:
            assert not hold & set(r["context_ids"]), r["id"]
            assert r["policy_id"] not in hold


def test_gold_labels_follow_business_rules(splits, corpus):
    for rows in splits.values():
        for r in rows:
            ctx = [corpus.by_id(i) for i in r["context_ids"]]
            assert len(ctx) == N_CONTEXT
            assert tuple(r["gold"].keys()) == ANSWER_KEYS
            assert r["gold"] == gold_answer(r["topic"], r["region_code"], ctx, corpus)
            target = resolve(r["topic"], r["region_code"], ctx)
            if r["is_unanswerable"]:
                assert target is None and r["gold"]["doc_code"] is None
            else:
                assert target.id == r["policy_id"], "văn bản đúng phải là văn bản gốc của kịch bản"
                others = [p for p in ctx if p is not target and applicable(p, r["topic"], r["region_code"])]
                assert all(not p.is_active and target.is_active for p in others), "văn bản nhiễu không được là đáp án hợp lệ"


def test_split_sizes_and_mix(splits):
    tr, te = dataset_stats(splits["train"]), dataset_stats(splits["test"])
    assert tr["n"] >= 250 and te["n"] >= 90
    assert 0.25 <= tr["unanswerable_ratio"] <= 0.45
    assert tr["colloquial_ratio"] == 0.5 and te["colloquial_ratio"] == 0.5
    assert te["slices"]["unseen_doc"] > 0 and te["slices"]["ood"] > 0
    assert sum(r["is_expired"] for r in splits["test"]) >= 5


def test_reasoning_mentions_every_context_doc(splits):
    for r in splits["train"][:80]:
        if not r["is_unanswerable"]:
            for pid_code in [r["gold"]["doc_code"]]:
                assert pid_code in r["gold"]["verification_reasoning"]
