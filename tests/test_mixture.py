"""混合样本解析：共迁移聚带与子集选定纯算法测试 + API 端到端测试。"""
from fastapi.testclient import TestClient

from app.main import app
from app.mixture import cluster_merged_bands, resolve_mixture
from app.models import MixtureRequest

client = TestClient(app)


def _req(**kw):
    payload = {
        "candidates": [
            {"name": "alpha", "bands": [100, 200]},
            {"name": "beta", "bands": [300, 400]},
            {"name": "gamma", "bands": [500, 600]},
        ],
        "observed_bands": [102, 198, 305, 395],
        "band_match_tolerance": 10,
        "max_components": 2,
    }
    payload.update(kw)
    return payload


# ---------------------------------------------------------------- 共迁移聚带（纯函数）

def test_cluster_comigrating_bands_merge_once_with_mean():
    # 200 与 205 共迁移（容差 10）：只出现一次，代表长度取均值
    assert cluster_merged_bands([205, 100, 200, 300], 10) == [100, 202.5, 300]


def test_cluster_single_linkage_chains_and_breaks():
    # 单连接：相邻差均 <= 容差时链式归入同一条带，即使首尾差超过容差
    assert cluster_merged_bands([100, 108, 116], 10) == [108]
    # 相邻差超过容差处断开成新带
    assert cluster_merged_bands([100, 108, 120], 10) == [104, 120]
    # 容差为 0 时仅完全等长的条带共迁移
    assert cluster_merged_bands([100, 100, 101], 0) == [100, 101]


def test_cluster_empty_bands():
    assert cluster_merged_bands([], 10) == []


# ---------------------------------------------------------------- 子集选定（纯函数）

def test_resolve_picks_true_pair_over_singletons():
    res = resolve_mixture(MixtureRequest(**_req()))
    assert res.selected_components == ["alpha", "beta"]
    assert res.merged_predicted_bands == [100, 200, 300, 400]
    assert res.match.matched_count == 4
    assert res.match.missing_count == 0 and res.match.extra_count == 0
    assert res.match.total_error == 14  # 2 + 2 + 5 + 5
    assert res.evaluated_subset_count == 6  # C(3,1) + C(3,2)
    assert res.candidate_count == 3 and res.observed_band_count == 4
    assert res.selection_reason


def test_resolve_comigrating_bands_merge_before_matching():
    res = resolve_mixture(MixtureRequest(**_req(
        candidates=[
            {"name": "alpha", "bands": [100, 200]},
            {"name": "beta", "bands": [205, 300]},
        ],
        observed_bands=[101, 203, 299],
    )))
    assert res.selected_components == ["alpha", "beta"]
    # 200 与 205 共迁移：合并为均值 202.5，只出现一次
    assert res.merged_predicted_bands == [100, 202.5, 300]
    assert res.match.matched_count == 3
    assert res.match.missing_count == 0 and res.match.extra_count == 0
    assert res.evaluated_subset_count == 3  # C(2,1) + C(2,2)


def test_resolve_prefers_fewer_components_on_tie():
    # 两候选条带完全相同：单成分即可达到同样的未匹配数与总误差
    res = resolve_mixture(MixtureRequest(**_req(
        candidates=[
            {"name": "beta", "bands": [100, 200]},
            {"name": "alpha", "bands": [100, 200]},
        ],
        observed_bands=[100, 200],
    )))
    # 成分数少者优先；单成分内部并列时按候选名字典序
    assert res.selected_components == ["alpha"]
    assert res.evaluated_subset_count == 3


def test_resolve_prefers_smaller_total_error_on_tie():
    res = resolve_mixture(MixtureRequest(**_req(
        candidates=[
            {"name": "alpha", "bands": [100]},
            {"name": "beta", "bands": [104]},
        ],
        observed_bands=[101],
        band_match_tolerance=5,
        max_components=1,
    )))
    # 两者未匹配条带数相同（各 0），alpha 总误差 1 < beta 3
    assert res.selected_components == ["alpha"]
    assert res.match.total_error == 1


def test_resolve_lexicographic_last_tiebreak():
    # 未匹配数、总误差、成分数全部并列时按候选名列表字典序
    res = resolve_mixture(MixtureRequest(**_req(
        candidates=[
            {"name": "zeta", "bands": [100]},
            {"name": "alpha", "bands": [100]},
        ],
        observed_bands=[100],
        max_components=1,
    )))
    assert res.selected_components == ["alpha"]


def test_resolve_candidate_order_does_not_matter():
    first = resolve_mixture(MixtureRequest(**_req())).model_dump()
    shuffled = _req(candidates=list(reversed(_req()["candidates"])))
    second = resolve_mixture(MixtureRequest(**shuffled)).model_dump()
    assert first == second


def test_resolve_max_components_clamped_to_candidate_count():
    # 上限超过候选数时按候选数计：评估全部非空子集
    res = resolve_mixture(MixtureRequest(**_req(max_components=99)))
    assert res.evaluated_subset_count == 7  # 2^3 - 1
    assert res.selected_components == ["alpha", "beta"]


def test_resolve_max_components_one_only_evaluates_singletons():
    res = resolve_mixture(MixtureRequest(**_req(max_components=1)))
    assert res.evaluated_subset_count == 3
    # 单成分中 alpha 未匹配 2 条（额外 305、395），总误差 4 最小
    assert res.selected_components == ["alpha"]
    assert res.match.extra_bands == [305, 395]


# ---------------------------------------------------------------- API 端到端

def test_mixture_api_basic():
    res = client.post("/api/v1/digest/mixture", json=_req())
    assert res.status_code == 200
    body = res.json()

    assert body["selected_components"] == ["alpha", "beta"]
    assert body["merged_predicted_bands"] == [100, 200, 300, 400]
    assert body["evaluated_subset_count"] == 6
    assert body["candidate_count"] == 3
    assert body["observed_band_count"] == 4
    assert body["band_match_tolerance"] == 10
    assert body["max_components"] == 2
    assert body["selection_reason"]

    match = body["match"]
    assert match["matched_count"] == 4
    assert match["missing_bands"] == [] and match["extra_bands"] == []
    assert match["total_error"] == 14
    assert [m["predicted_length"] for m in match["matches"]] == [100, 200, 300, 400]
    for m in match["matches"]:
        assert m["error"] == abs(m["predicted_length"] - m["observed_length"])
        assert m["description"]
    # 计数一致性
    assert match["matched_count"] == len(match["matches"])
    assert match["matched_count"] + match["missing_count"] == match["predicted_band_count"]
    assert match["matched_count"] + match["extra_count"] == body["observed_band_count"]


def test_mixture_api_reports_missing_and_extra():
    body = client.post("/api/v1/digest/mixture", json=_req(
        observed_bands=[102, 198, 305, 395, 900],
        max_components=1,
    )).json()
    # 单成分上限：alpha 缺失 0 条、额外 3 条（305/395/900），总误差 4 最小
    assert body["selected_components"] == ["alpha"]
    assert body["match"]["missing_bands"] == []
    assert body["match"]["extra_bands"] == [305, 395, 900]
    assert body["match"]["missing_count"] == 0
    assert body["match"]["extra_count"] == 3
    assert body["evaluated_subset_count"] == 3


def test_mixture_api_float_bands_and_cluster_mean_rounding():
    body = client.post("/api/v1/digest/mixture", json=_req(
        candidates=[
            {"name": "alpha", "bands": [100, 200.4]},
            {"name": "beta", "bands": [199.6, 300]},
        ],
        observed_bands=[100.4, 200, 299.6],
        band_match_tolerance=1,
    )).json()
    assert body["selected_components"] == ["alpha", "beta"]
    # 200.4 与 199.6 共迁移，代表长度取均值 200.0
    assert body["merged_predicted_bands"] == [100, 200, 300]
    assert body["match"]["matched_count"] == 3
    assert body["match"]["total_error"] == 0.8


# ---------------------------------------------------------------- mixture 422 校验

def _expect_mixture_422(payload, code_part=None):
    res = client.post("/api/v1/digest/mixture", json=payload)
    assert res.status_code == 422
    body = res.json()
    assert body["code"] == "validation_failed"
    assert body["errors"]
    if code_part:
        assert any(code_part in e["type"] for e in body["errors"])
    return body


def test_mixture_422_invalid_max_components():
    _expect_mixture_422(_req(max_components=0))
    _expect_mixture_422(_req(max_components=-2))
    _expect_mixture_422(_req(max_components="two"))
    payload = _req()
    del payload["max_components"]
    _expect_mixture_422(payload)


def test_mixture_422_duplicate_candidate_name():
    _expect_mixture_422(_req(candidates=[
        {"name": "alpha", "bands": [100]},
        {"name": "alpha", "bands": [200]},
    ]), "duplicate_candidate_name")


def test_mixture_422_blank_candidate_name():
    _expect_mixture_422(
        _req(candidates=[{"name": "  ", "bands": [100]}]), "invalid_candidate_name",
    )
    _expect_mixture_422(_req(candidates=[{"name": "", "bands": [100]}]))


def test_mixture_422_empty_candidates():
    # 候选列表为空
    body = _expect_mixture_422(_req(candidates=[]))
    assert any("candidates" in e["loc"] for e in body["errors"])
    # 候选的预测条带列表为空
    body = _expect_mixture_422(
        _req(candidates=[{"name": "alpha", "bands": []}]), "empty_candidate_bands",
    )
    assert any("bands" in e["loc"] for e in body["errors"])


def test_mixture_422_too_many_candidates(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_MIXTURE_CANDIDATES", 3)
    payload = _req(candidates=[
        {"name": n, "bands": [100]} for n in ("a", "b", "c", "d")
    ])
    _expect_mixture_422(payload, "too_many_candidates")
    # 恰好等于上限时放行
    assert client.post("/api/v1/digest/mixture", json=_req()).status_code == 200


def test_mixture_422_exceeds_default_candidate_limit():
    payload = _req(candidates=[
        {"name": f"c{i}", "bands": [100]} for i in range(13)
    ])
    _expect_mixture_422(payload, "too_many_candidates")


def test_mixture_422_empty_observed():
    body = _expect_mixture_422(_req(observed_bands=[]))
    assert any("observed_bands" in e["loc"] for e in body["errors"])


def test_mixture_422_negative_band_and_invalid_tolerance():
    _expect_mixture_422(_req(observed_bands=[100, -5]))
    _expect_mixture_422(_req(candidates=[{"name": "alpha", "bands": [-1]}]))
    _expect_mixture_422(_req(band_match_tolerance=-1))
    _expect_mixture_422(_req(band_match_tolerance="wide"))


def test_mixture_422_missing_required_fields():
    _expect_mixture_422({"observed_bands": [100], "band_match_tolerance": 10,
                         "max_components": 1})
    _expect_mixture_422({})


def test_mixture_422_too_many_bands(monkeypatch):
    from app.config import settings
    monkeypatch.setattr(settings, "MAX_BANDS", 4)
    _expect_mixture_422(_req(observed_bands=[1, 2, 3, 4, 5]), "too_many_bands")
    _expect_mixture_422(
        _req(candidates=[{"name": "alpha", "bands": [1, 2, 3, 4, 5]}]),
        "too_many_bands",
    )
