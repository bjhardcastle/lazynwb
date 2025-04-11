class ColumnError(KeyError):
    """Requested column name is not in table group"""

    pass


class InternalPathError(KeyError):
    """Requested internal path is not in file"""

    pass
