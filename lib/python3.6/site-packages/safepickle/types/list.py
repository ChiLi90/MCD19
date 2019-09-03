from .interface import EncoderDecoderType


class ListType(EncoderDecoderType):
    def can_encode(self, obj):
        return isinstance(obj, list)

    def encode(self, obj, encode_fn):
        return [encode_fn(val) for val in obj]

    def can_decode(self, obj):
        pass

    def decode(self, obj):
        # no special decoding needed
        pass

    def get_type(self):
        return list
