"""
Benchmark different SELFIES tokenization methods.
"""

import time
from selfies_tokenizer import SELFIESTokenizer

# Test cases with varying complexity
TEST_CASES = [
    '[C][=O]',
    '[C][C][C][C][C][C]',
    '[C][=C][C][=C][C][=C][Ring1][Branch1_1]',
    '[C][C][Branch1][C][O][C][C][Branch1][C][O][C][C][Branch1][C][O][C]',
    '[N][C][=Branch1][C][=O][C][C][C][C][N][C][=Branch1][C][=O][O][C][Branch1][C][C][Branch1][C][C][C]',
]

# Longer test case for stress testing
LONG_SELFIES = '[C][C][C][C][C]' * 100


def benchmark_method(method_name: str, selfies_string: str, iterations: int = 10000):
    """Benchmark a specific tokenization method."""
    tokenizer = SELFIESTokenizer(method=method_name)

    start_time = time.perf_counter()
    for _ in range(iterations):
        tokens = tokenizer.tokenize(selfies_string)
    end_time = time.perf_counter()

    elapsed = end_time - start_time
    per_call = (elapsed / iterations) * 1_000_000  # Convert to microseconds

    return elapsed, per_call, tokens


def run_benchmarks():
    """Run benchmarks for all methods."""
    methods = ['regex', 'manual']

    try:
        import selfies
        methods.append('selfies_lib')
    except ImportError:
        print("Note: selfies library not installed, skipping selfies_lib benchmark\n")

    print("SELFIES Tokenizer Benchmark")
    print("=" * 80)

    for test_case in TEST_CASES[:3]:  # Test first 3 cases
        print(f"\nTest case: {test_case}")
        print("-" * 80)

        results = []
        for method in methods:
            try:
                elapsed, per_call, tokens = benchmark_method(method, test_case)
                results.append((method, elapsed, per_call))
                print(f"{method:15s}: {elapsed:.4f}s total, {per_call:.2f}µs per call")
                print(f"                 Result: {tokens}")
            except Exception as e:
                print(f"{method:15s}: ERROR - {e}")

        # Show the winner
        if results:
            winner = min(results, key=lambda x: x[1])
            print(f"\nFastest: {winner[0]} ({winner[2]:.2f}µs per call)")

    # Stress test with long SELFIES
    print("\n" + "=" * 80)
    print(f"Stress test with long SELFIES ({len(LONG_SELFIES)} chars)")
    print("-" * 80)

    results = []
    for method in methods:
        try:
            elapsed, per_call, tokens = benchmark_method(method, LONG_SELFIES, iterations=1000)
            results.append((method, elapsed, per_call))
            print(f"{method:15s}: {elapsed:.4f}s total, {per_call:.2f}µs per call")
            print(f"                 Tokens: {len(tokens)}")
        except Exception as e:
            print(f"{method:15s}: ERROR - {e}")

    if results:
        winner = min(results, key=lambda x: x[1])
        print(f"\nFastest: {winner[0]} ({winner[2]:.2f}µs per call)")


if __name__ == '__main__':
    run_benchmarks()