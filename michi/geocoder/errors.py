class IntersectionNotFoundError(ValueError):
    pass

class StreetNameNotFoundError(KeyError):
    def __init__(self, street, options):
        self.street = street
        self.options = options

        super().__init__(""" %s not found. See `StreetNameNotFoundError.options`
            for possible matches.""" % street)

class NotAStreetError(ValueError):
    def __init__(self, street, street_code):
        self.street = street
        self.street_code = street_code

        super().__init__("%s is not a street." % self.street)

class StreetStretchNotFoundError(ValueError):
    pass

class WrongWayError(ValueError):
    pass
