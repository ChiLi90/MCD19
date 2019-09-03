from .interface import EncoderDecoderType


class InstanceType(EncoderDecoderType):
    def can_encode(self, obj):
        return hasattr(obj, "__dict__")

    def encode(self, obj, encode_fn):
        return encode_fn(obj.__dict__)

    def can_decode(self, obj):
        pass

    def decode(self, obj):
        # no special decoding needed
        pass

    def get_type(self):
        return object
