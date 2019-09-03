import json

from .encoding import encode, decode
from pickle import PicklingError, UnpicklingError


def load(file):
    """ Read a json object representation from the opened file object file. 

    Args:
        file: file object

    Returns:
        object representation as a dict
    """
    try:
        return json.load(file, object_hook=decode)
    except (TypeError, ValueError) as e:
        raise UnpicklingError(str(e))


def dump(obj, file, **kwargs):
    """ Write a json representation to the open file object

    Args:
        obj: object to represent  
        file: open file object 
    """
    # apply json defaults if not present
    if 'indent' not in kwargs:
        kwargs['indent'] = 4
    if 'separators' not in kwargs:
        kwargs['separators'] = (',', ': ')

    try:
        json.dump(encode(obj), file, **kwargs)
    except TypeError as e:
        raise PicklingError(str(e))


def loads(bytes_obj, encoding="utf-8", errors="strict"):
    """ Read a json object representation from the bytes representation. 

    Args:
        bytes_obj (bytes): bytes object representation
        encoding (str): encoding to use to decode bytes
        errors (str): same as decode 'errors' argument.

    Returns:
        object representation as a dict
    """
    str_obj = bytes_obj.decode(encoding=encoding, errors=errors)
    try:
        return json.loads(str_obj, object_hook=decode)
    except ValueError as e:
        raise UnpicklingError(str(e))


def dumps(obj, encoding="utf-8", errors="strict"):
    """ Write a json representation of object as a bytes object

    Args:
        obj: object to represent
        encoding (str): encoding to use to encode bytes
        errors (str): same as encode 'errors' argument.

    Returns:
        object representation as a bytes object
    """
    try:
        str_obj = json.dumps(encode(obj))
    except TypeError as e:
        raise PicklingError(str(e))

    return str_obj.encode(encoding=encoding, errors=errors)
