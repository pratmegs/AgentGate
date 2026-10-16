from src.evaluator import run, metrics


def test_benchmark(tmp_path):
    output = run(output_path=tmp_path / 'evaluation.json')
    assert output['metrics']['case_count'] >= 30
    assert output['metrics']['unsafe_autonomy_rate'] == 0
    assert output['metrics']['overall_decision_accuracy'] == 1


def test_empty_metrics():
    assert metrics([])['unsafe_autonomy_rate'] is None
