from .interface import EncoderDecoderType


class ByteArrayType(EncoderDecoderType):
    def can_encode(self, obj):
        return isinstance(obj, bytearray)

    def encode(self, obj, encode_fn):
        return {'__bytearray__': [encode_fn(val) for val in obj]}

    def can_decode(self, obj):
        return isinstance(obj, dict) and '__bytearray__' in obj

    def decode(self, obj):
        return bytearray(obj['__bytearray__'])

    def get_type(self):
        return bytearray
