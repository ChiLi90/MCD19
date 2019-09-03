from .interface import EncoderDecoderType


class BytesType(EncoderDecoderType):
    def can_encode(self, obj):
        return isinstance(obj, bytes)

    def encode(self, obj, encode_fn):
        return {'__bytes__': [encode_fn(val) for val in obj]}

    def can_decode(self, obj):
        return isinstance(obj, dict) and '__bytes__' in obj

    def decode(self, obj):
        return bytes(obj['__bytes__'])

    def get_type(self):
        return bytes
