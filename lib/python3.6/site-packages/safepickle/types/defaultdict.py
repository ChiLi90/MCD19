from collections import defaultdict

from .interface import EncoderDecoderType


class DefaultDictType(EncoderDecoderType):
    def can_encode(self, obj):
        return isinstance(obj, defaultdict)

    def encode(self, obj, encode_fn):
        from .manager import TypesManager
        return {"__defaultdict__":
                [(encode_fn(k), encode_fn(v)) for k, v in obj.items()],
                "type": TypesManager.get_str_type(obj.default_factory)}

    def can_decode(self, obj):
        return isinstance(obj, dict) and '__defaultdict__' in obj

    def decode(self, obj):
        from .manager import TypesManager
        obj_as_dict = {k: v for (k, v) in obj['__defaultdict__']}
        return defaultdict(TypesManager.get_type(obj['type']), obj_as_dict)

    def get_type(self):
        return defaultdict
