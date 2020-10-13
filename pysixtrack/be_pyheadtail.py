from .base_classes import Element


class PyHEADTAILInterface(Element):
    _description = [
        ("direction", "", "0 == to PyHEADTAIL, 1 == from PyHEADTAIL", 0)
    ]

    def track(self, p):
        pass
