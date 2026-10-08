from .types import TypeEncoder

def test_type_encoder_dimension():
    encoder = TypeEncoder()
    assert encoder.dimension == 2
