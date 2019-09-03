from .interface import EncoderDecoderType


class TupleType(EncoderDecoderType):
    def can_encode(self, obj):
        return isinstance(obj, tuple)

    def encode(self, obj, encode_fn):
        return {'__tuple__': [encode_fn(val) for val in obj]}

    def can_decode(self, obj):
        return isinstance(obj, dict) and '__tuple__' in obj

    def decode(self, obj):
        return tuple(obj['__tuple__'])

    def get_type(self):
        return tuple
