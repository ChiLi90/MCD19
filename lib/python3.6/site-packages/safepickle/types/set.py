from .interface import EncoderDecoderType


class SetType(EncoderDecoderType):
    def can_encode(self, obj):
        return isinstance(obj, set)

    def encode(self, obj, encode_fn):
        return {'__set__': [encode_fn(val) for val in obj]}

    def can_decode(self, obj):
        return isinstance(obj, dict) and '__set__' in obj

    def decode(self, obj):
        return set(obj['__set__'])

    def get_type(self):
        return set
