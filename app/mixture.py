"""核心算法：混合样本解析——候选子集枚举、共迁移聚带与最优子集选定。

所有函数均为纯函数，不依赖任何外部服务或全局状态。

混合模型
--------
观测条带来自不超过 ``max_components`` 个候选样本的混合：混合样本中
长度差不超过匹配容差的条带共迁移为同一条带（只出现一次），因此先把
子集内全部候选的预测条带合并，按容差做单连接聚带（与
:mod:`app.digest` 的聚带规则一致），每条聚带以成员长度均值为
代表长度，得到该子集的合并预测条带。

子集评估与选定
--------------
对每个不超过上限的候选子集，复用 :mod:`app.identify` 的一对一最小
代价匹配，将合并预测条带与观测条带配对，统计未匹配条带数
（缺失 + 额外）与总误差。全部子集按

  未匹配条带总数升序 → 总误差升序 → 成分数升序 → 候选名列表字典序

的固定顺序比较，首个即为唯一选定结果：候选名互不相同，完整排序键
必然唯一，结果与候选的提交顺序无关。

子集数量为 sum(C(n, k))（k = 1..max_components），候选数 n 不超过
MAX_MIXTURE_CANDIDATES（默认 12），规模可控。
"""
from itertools import combinations

from app.identify import match_sample
from app.models import MixtureRequest, MixtureResponse, SampleMatchOut


def cluster_merged_bands(bands: list[float], tolerance: float) -> list[float]:
    """把多条预测条带按容差做单连接聚带：共迁移条带只出现一次，代表长度取均值。"""
    clusters: list[list[float]] = []
    for length in sorted(bands):
        if clusters and length - clusters[-1][-1] <= tolerance:
            clusters[-1].append(length)
        else:
            clusters.append([length])
    return [round(sum(cluster) / len(cluster), 2) for cluster in clusters]


def _selection_key(match: SampleMatchOut, names: tuple[str, ...]) -> tuple:
    """子集排序键：未匹配条带总数 → 总误差 → 成分数 → 候选名列表字典序。"""
    return (
        match.missing_count + match.extra_count,
        match.total_error,
        len(names),
        names,
    )


def resolve_mixture(req: MixtureRequest) -> MixtureResponse:
    """枚举全部不超过成分上限的候选子集，选出唯一最优成分组合。"""
    names = sorted(candidate.name for candidate in req.candidates)
    bands_by_name = {candidate.name: list(candidate.bands) for candidate in req.candidates}
    observed = list(req.observed_bands)
    max_size = min(req.max_components, len(names))

    best_key: tuple | None = None
    best_names: tuple[str, ...] = ()
    best_merged: list[float] = []
    best_match: SampleMatchOut | None = None
    evaluated = 0
    for size in range(1, max_size + 1):
        for combo in combinations(names, size):
            merged = cluster_merged_bands(
                [band for name in combo for band in bands_by_name[name]],
                req.band_match_tolerance,
            )
            match = match_sample(merged, observed, req.band_match_tolerance)
            evaluated += 1
            key = _selection_key(match, combo)
            if best_key is None or key < best_key:
                best_key = key
                best_names, best_merged, best_match = combo, merged, match

    assert best_match is not None  # 至少 1 个候选且上限 >= 1，必然评估过子集
    unmatched = best_match.missing_count + best_match.extra_count
    reason = (
        f"共评估 {evaluated} 个候选子集；成分 {'+'.join(best_names)} 的未匹配条带 "
        f"{unmatched} 条（缺失 {best_match.missing_count} 条、额外 {best_match.extra_count} 条）"
        f"最少，总误差 {best_match.total_error:g} bp；并列时依次按成分数升序、"
        f"候选名列表字典序选定"
    )
    return MixtureResponse(
        band_match_tolerance=req.band_match_tolerance,
        max_components=req.max_components,
        candidate_count=len(names),
        observed_band_count=len(observed),
        evaluated_subset_count=evaluated,
        selected_components=list(best_names),
        merged_predicted_bands=best_merged,
        match=best_match,
        selection_reason=reason,
    )
