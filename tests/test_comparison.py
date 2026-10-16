from src.comparison import model_prediction
import pytest


@pytest.mark.parametrize('authority,approval,autonomous,expected', [
    ('YES', 'NO', 'YES', 'ALLOW'), ('NO', 'NO', 'YES', 'BLOCK'),
    ('YES', 'YES', 'YES', 'REQUIRE_APPROVAL'), ('YES', 'NO', 'NO', 'REQUIRE_APPROVAL')])
def test_mapping(authority, approval, autonomous, expected):
    answers = {k: {'choice': v} for k, v in dict(within_authority=authority, human_approval=approval, autonomous=autonomous, consequence='HIGH').items()}
    assert model_prediction({'status': 'available', 'result': {'answers': answers}}) == (expected, 'HIGH')


def test_unavailable_is_not_a_prediction():
    assert model_prediction({'status': 'unavailable'}) is None
