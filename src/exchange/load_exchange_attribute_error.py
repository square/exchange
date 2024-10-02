class LoadExchangeAttributeError(Exception):
    def __init__(self, attribute_name: str, attribute_value: str) -> None:
        self.attribute_name = attribute_name
        self.attribute_value = attribute_value
        self.message = f"Unknown {attribute_name}: {attribute_value}"
        super().__init__(self.message)
