class PineconeException(Exception):
    """The base exception class for all exceptions in the Pinecone Python SDK"""


class PineconeApiTypeError(PineconeException, TypeError):
    def __init__(self, msg, path_to_item=None, valid_classes=None, key_type=None) -> None:
        """Raises an exception for TypeErrors

        Args:
            msg (str): the exception message

        Keyword Args:
            path_to_item (list): a list of keys an indices to get to the
                                 current_item
                                 None if unset
            valid_classes (tuple): the primitive classes that current item
                                   should be an instance of
                                   None if unset
            key_type (bool): False if our value is a value in a dict
                             True if it is a key in a dict
                             False if our item is an item in a list
                             None if unset
        """
        self.path_to_item = path_to_item
        self.valid_classes = valid_classes
        self.key_type = key_type
        full_msg = msg
        if path_to_item:
            full_msg = "{0} at {1}".format(msg, render_path(path_to_item))
        super(PineconeApiTypeError, self).__init__(full_msg)


class PineconeApiValueError(PineconeException, ValueError):
    def __init__(self, msg, path_to_item=None) -> None:
        """
        Args:
            msg (str): the exception message

        Keyword Args:
            path_to_item (list) the path to the exception in the
                received_data dict. None if unset
        """

        self.path_to_item = path_to_item
        full_msg = msg
        if path_to_item:
            full_msg = "{0} at {1}".format(msg, render_path(path_to_item))
        super(PineconeApiValueError, self).__init__(full_msg)


class PineconeApiAttributeError(PineconeException, AttributeError):
    """Raised when an attribute reference or assignment fails."""

    def __init__(self, msg, path_to_item=None) -> None:
        """
        Args:
            msg (str): the exception message

        Keyword Args:
            path_to_item (None/list) the path to the exception in the
                received_data dict
        """
        self.path_to_item = path_to_item
        full_msg = msg
        if path_to_item:
            full_msg = "{0} at {1}".format(msg, render_path(path_to_item))
        super(PineconeApiAttributeError, self).__init__(full_msg)


class PineconeApiKeyError(PineconeException, KeyError):
    def __init__(self, msg, path_to_item=None) -> None:
        """
        Args:
            msg (str): the exception message

        Keyword Args:
            path_to_item (None/list) the path to the exception in the
                received_data dict
        """
        self.path_to_item = path_to_item
        full_msg = msg
        if path_to_item:
            full_msg = "{0} at {1}".format(msg, render_path(path_to_item))
        super(PineconeApiKeyError, self).__init__(full_msg)


class PineconeApiException(PineconeException):
    """Raised when an API exception occurs."""

    def __init__(self, status=None, reason=None, http_resp=None) -> None:
        if http_resp:
            self.status = http_resp.status
            self.reason = http_resp.reason
            self.body = http_resp.data
            self.headers = http_resp.getheaders()
        else:
            self.status = status
            self.reason = reason
            self.body = None
            self.headers = None

    def __str__(self):
        """Custom error messages for exception"""
        error_message = "({0})\nReason: {1}\n".format(self.status, self.reason)
        if self.headers:
            error_message += "HTTP response headers: {0}\n".format(self.headers)

        if self.body:
            error_message += "HTTP response body: {0}\n".format(self.body)

        return error_message


class NotFoundException(PineconeApiException):
    """Raised when a resource is not found."""

    def __init__(self, status=None, reason=None, http_resp=None) -> None:
        super(NotFoundException, self).__init__(status, reason, http_resp)


class UnauthorizedException(PineconeApiException):
    """Raised when access to a resource is not authorized."""

    def __init__(self, status=None, reason=None, http_resp=None) -> None:
        super(UnauthorizedException, self).__init__(status, reason, http_resp)


class ForbiddenException(PineconeApiException):
    """Raised when access to a resource is forbidden."""

    def __init__(self, status=None, reason=None, http_resp=None) -> None:
        super(ForbiddenException, self).__init__(status, reason, http_resp)


class ServiceException(PineconeApiException):
    """Raised when a service error occurs."""

    def __init__(self, status=None, reason=None, http_resp=None) -> None:
        super(ServiceException, self).__init__(status, reason, http_resp)


def render_path(path_to_item):
    """Returns a string representation of a path"""
    result = ""
    for pth in path_to_item:
        if isinstance(pth, int):
            result += "[{0}]".format(pth)
        else:
            result += "['{0}']".format(pth)
    return result


class PineconeProtocolError(PineconeException):
    """Raised when something unexpected happens mid-request/response."""


class PineconeConfigurationError(PineconeException):
    """Raised when a configuration error occurs."""


class ListConversionException(PineconeException, TypeError):
    def __init__(self, message):
        super().__init__(message)


__all__ = [
    "PineconeException",
    "PineconeApiTypeError",
    "PineconeApiValueError",
    "PineconeApiAttributeError",
    "PineconeApiKeyError",
    "PineconeApiException",
    "NotFoundException",
    "UnauthorizedException",
    "ForbiddenException",
    "ServiceException",
    "PineconeProtocolError",
    "PineconeConfigurationError",
    "ListConversionException",
]
