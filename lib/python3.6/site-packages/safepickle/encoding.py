from .types import TypesManager


def encode(obj):
    """ Encodes an item preparing it to be json serializable
    
    Encode relies on defined custom types to provide encoding, which in turn 
    are responsible of using the 'encode' function parameter passed to them
    to recursively encoded contained items.
    
    Args:
        obj: item to encode
         
    Returns:
        encoded item
    """

    # handle basic types separately
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    for type_ in TypesManager.get_types():
        if type_.can_encode(obj):
            return type_.encode(obj, encode)

    raise TypeError("Type: '{}' is not supported".format(type(obj)))


def decode(dct):
    """ Object hook to use to decode object literals.
    
    This function is called from within json loading mechanism 
    for every literal
    
    Args:
        dct (dict):  literal to decode

    Returns:
         decoded literal
    """
    for type_ in TypesManager.get_types():
        if type_.can_decode(dct):
            return type_.decode(dct)

    return dct
