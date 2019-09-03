from datetime import datetime

from .interface import EncoderDecoderType


class DatetimeType(EncoderDecoderType):
    def can_encode(self, obj):
        return isinstance(obj, datetime)

    def encode(self, obj, _):
        return {'__datetime__': {'year': obj.year,
                                 'month': obj.month,
                                 'day': obj.day,
                                 'hour': obj.hour,
                                 'minute': obj.minute,
                                 'second': obj.second,
                                 'microsecond': obj.microsecond}}

    def can_decode(self, obj):
        return isinstance(obj, dict) and '__datetime__' in obj

    def decode(self, obj):
        return datetime(**obj['__datetime__'])

    def get_type(self):
        return datetime
