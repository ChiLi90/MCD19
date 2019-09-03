class EncoderDecoderType(object):
    def can_encode(self, obj):
        """ Defines if object is to be encoded by this type

        Args:
            obj: instance to test against 

        Returns:
            True if type can handle instance, False otherwise
        """
        raise NotImplementedError

    def encode(self, obj, encode_fn):
        """ Provides encoding for given instance 

        Args:
            obj: instance to encode 
            encode_fn (fn): function to use to encode recursively
                potential types within obj
        """
        raise NotImplementedError

    def can_decode(self, obj):
        """ Defines if object is to be decoded by this type

        Args:
            obj: instance to test against 

        Returns:
            True if type can handle instance, False otherwise
        """
        raise NotImplementedError

    def decode(self, obj):
        """ Provides decoding for given instance 

        Args:
            obj: instance to decode 
        """
        raise NotImplementedError

    def get_type(self):
        """ Provides supported type
        """
        raise NotImplementedError
