import numpy as np
import pandas as pd
import pytest

from eegscore.features import BaselineStats, baseline_normalise


def test_baseline_normalise_uses_only_baseline_rows():
    rng = np.random.default_rng(0)
    groups = np.repeat([1, 2], 20)
    base = np.r_[np.ones(5, bool), np.zeros(15, bool), np.ones(5, bool), np.zeros(15, bool)]
    F = pd.DataFrame({"f": rng.standard_normal(40)})
    F.loc[groups == 2, "f"] += 100.0                    # huge subject offset
    Z = baseline_normalise(F, groups, base)
    for g in (1, 2):
        m = (groups == g) & base
        assert abs(Z.loc[m, "f"].mean()) < 1e-9        # baseline rows are centred
        assert abs(Z.loc[m, "f"].std(ddof=1) - 1) < 1e-6
    # subject offset removed for scored rows too
    assert abs(Z.loc[(groups == 1) & ~base, "f"].mean() - Z.loc[(groups == 2) & ~base, "f"].mean()) < 3


def test_baseline_stats_roundtrip_and_min_windows():
    F = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [0.0, 0.0, 0.0]})
    st = BaselineStats.fit(F)
    d = st.to_dict()
    st2 = BaselineStats.from_dict(d)
    assert np.allclose(st2.transform(F).to_numpy(), st.transform(F).to_numpy())
    assert np.isfinite(st.transform(F).to_numpy()).all()   # zero-variance feature guarded
    with pytest.raises(ValueError):
        BaselineStats.fit(F.iloc[:2])
