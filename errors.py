class InvalidGDSFormatError(Exception):
    def __init__(self, message):
        super(InvalidGDSFormatError, self).__init__(message)

class PFComputationError(Exception):
    def __init__(self, message):
        super(PFComputationError, self).__init__(message)

class PowerIterationError(Exception):
    def __init__(self, message):
        super(PowerIterationError, self).__init__(message)
