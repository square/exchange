from exchange.load_exchange_attribute_error import LoadExchangeAttributeError


def test_load_exchange_attribute_error():
    attribute_name = "provider"
    attribute_value = "not_exist"
    error = LoadExchangeAttributeError(attribute_name, attribute_value)

    assert error.attribute_name == attribute_name
    assert error.attribute_value == attribute_value
    assert error.message == "Unknown provider: not_exist"
