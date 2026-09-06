"""Demonstrate order-flow-imbalance analysis on a synthetic order book.

Runs the same analysis on two datasets: one with a real, injected OFI effect
and one null dataset (ofi_impact=0) with no effect. A correct analysis
should find significance in the first and not the second — this is the
sanity check that the methodology isn't just finding noise.

Usage:
    python -m analysis.run_microstructure_analysis
"""
from analysis.microstructure import summarize
from analysis.orderbook_sim import simulate_order_book


def report(label: str, results: dict) -> None:
    print(f"\n=== {label} ===")
    for key, value in results.items():
        print(f"  {key:20s}: {value:.6f}" if isinstance(value, float) else f"  {key:20s}: {value}")
    if results["p_value"] < 0.05:
        print("  -> Significant OFI effect detected (p < 0.05)")
    else:
        print("  -> No significant OFI effect (correctly found nothing, as expected)")


def main():
    book_with_signal = simulate_order_book(ofi_impact=0.02, noise_std=0.2)
    book_null = simulate_order_book(ofi_impact=0.0, noise_std=0.2, seed=8)

    report("WITH INJECTED OFI EFFECT", summarize(book_with_signal))
    report("NULL (NO EFFECT) CONTROL", summarize(book_null))


if __name__ == "__main__":
    main()
