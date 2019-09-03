from .interface import EncoderDecoderType


class DictType(EncoderDecoderType):
    def can_encode(self, obj):
        return isinstance(obj, dict)

    def encode(self, obj, encode_fn):
        if all(isinstance(k, str) for k in obj):
            return {k: encode_fn(v) for k, v in obj.items()}
        return {"__dct__":
                [(encode_fn(k), encode_fn(v)) for k, v in obj.items()]}

    def can_decode(self, obj):
        return isinstance(obj, dict) and '__dct__' in obj

    def decode(self, obj):
        # try/catch can be removed when it is a fact that
        # there are no legacy files outstanding
        try:
            return {k: v for (k, v) in obj['__dct__']}
        except ValueError:
            # attempt to load it as legacy version used to
            return obj['__dct__']

    def get_type(self):
        return dict
