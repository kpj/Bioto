class InvalidGDSFormatError(Exception):
    def __init__(self, message):
        super(InvalidGDSFormatError, self).__init__(message)
