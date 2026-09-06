from analysis.microstructure import ofi_predictive_regression
from analysis.orderbook_sim import simulate_order_book


def test_injected_ofi_effect_is_detected_as_significant():
    book = simulate_order_book(ofi_impact=0.02, noise_std=0.2, n_ticks=5000)
    result = ofi_predictive_regression(book)
    assert result["p_value"] < 0.05
    assert result["slope"] > 0
    assert result["r_squared"] < 0.1  # real microstructure effects are small, not near-perfect fits


def test_null_dataset_shows_no_significant_effect():
    book = simulate_order_book(ofi_impact=0.0, noise_std=0.2, n_ticks=5000, seed=99)
    result = ofi_predictive_regression(book)
    assert result["p_value"] > 0.05


def test_simulated_book_has_expected_columns():
    book = simulate_order_book(n_ticks=100)
    expected = {"mid_price", "bid_price", "ask_price", "bid_volume", "ask_volume", "order_flow_imbalance"}
    assert expected.issubset(book.columns)
    assert (book["ask_price"] > book["bid_price"]).all()
