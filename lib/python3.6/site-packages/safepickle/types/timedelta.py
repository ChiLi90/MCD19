from datetime import timedelta

from .interface import EncoderDecoderType


class TimedeltaType(EncoderDecoderType):

    def can_encode(self, obj):
        return isinstance(obj, timedelta)

    def encode(self, obj, _):
        return {'__timedelta__': {'days': obj.days,
                                  'seconds': obj.seconds,
                                  'microseconds': obj.microseconds}}

    def can_decode(self, obj):
        return isinstance(obj, dict) and '__timedelta__' in obj

    def decode(self, obj):
        return timedelta(**obj['__timedelta__'])

    def get_type(self):
        return timedelta
