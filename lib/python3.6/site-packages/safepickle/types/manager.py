from .defaultdict import DefaultDictType
from .set import SetType
from .tuple import TupleType
from .dict import DictType
from .list import ListType
from .instance import InstanceType
from .datetime_ import DatetimeType
from .timedelta import TimedeltaType
from .bytearray import ByteArrayType
from .bytes import BytesType


class _TypesManager(object):
    """ Allows the definition of the types supported by the package
    """
    def __init__(self):
        self._types = [
            SetType(),
            TupleType(),
            DefaultDictType(),
            DictType(),
            ListType(),
            DatetimeType(),
            TimedeltaType(),
            ByteArrayType(),
            BytesType(),
            InstanceType()
        ]

    def get_types(self):
        return self._types

    def get_str_type(self, raw_type):
        """ Provides the string name for a given type
        """
        for type_ in (bool, int, float, str):
            if type_ == raw_type:
                return type_.__name__

        for type_ in self._types:
            if type_.get_type() == raw_type:
                return type_.__class__.__name__

    def get_type(self, str_type):
        """ Provides a type from its string name
        """
        for type_ in (bool, int, float, str):
            if str_type == type_.__name__:
                return type_

        for type_ in self._types:
            if type_.__class__.__name__ == str_type:
                return type_.get_type()


# Singleton instance to _TypesManager
TypesManager = _TypesManager()
