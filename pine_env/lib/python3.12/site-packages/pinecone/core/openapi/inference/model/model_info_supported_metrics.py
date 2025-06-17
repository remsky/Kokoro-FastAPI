"""
Pinecone Inference API

Pinecone is a vector database that makes it easy to search and retrieve billions of high-dimensional vectors.  # noqa: E501

This file is @generated using OpenAPI.

The version of the OpenAPI document: 2025-04
Contact: support@pinecone.io
"""

from pinecone.openapi_support.model_utils import (  # noqa: F401
    PineconeApiTypeError,
    ModelComposed,
    ModelNormal,
    ModelSimple,
    OpenApiModel,
    cached_property,
    change_keys_js_to_python,
    convert_js_args_to_python_args,
    date,
    datetime,
    file_type,
    none_type,
    validate_get_composed_info,
)
from pinecone.openapi_support.exceptions import PineconeApiAttributeError


def lazy_import():
    from pinecone.core.openapi.inference.model.model_info_metric import ModelInfoMetric

    globals()["ModelInfoMetric"] = ModelInfoMetric


from typing import Dict, Literal, Tuple, Set, Any, Type, TypeVar
from pinecone.openapi_support import PropertyValidationTypedDict, cached_class_property

T = TypeVar("T", bound="ModelInfoSupportedMetrics")


class ModelInfoSupportedMetrics(ModelSimple):
    """NOTE: This class is @generated using OpenAPI.

    Do not edit the class manually.

    Attributes:
      allowed_values (dict): The key is the tuple path to the attribute
          and the for var_name this is (var_name,). The value is a dict
          with a capitalized key describing the allowed value and an allowed
          value. These dicts store the allowed enum values.
      validations (dict): The key is the tuple path to the attribute
          and the for var_name this is (var_name,). The value is a dict
          that stores validations for max_length, min_length, max_items,
          min_items, exclusive_maximum, inclusive_maximum, exclusive_minimum,
          inclusive_minimum, and regex.
      additional_properties_type (tuple): A tuple of classes accepted
          as additional properties values.
    """

    _data_store: Dict[str, Any]
    _check_type: bool

    allowed_values: Dict[Tuple[str, ...], Dict[str, Any]] = {}

    validations: Dict[Tuple[str, ...], PropertyValidationTypedDict] = {}

    @cached_class_property
    def additional_properties_type(cls):
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded
        """
        lazy_import()
        return (bool, dict, float, int, list, str, none_type)  # noqa: E501

    _nullable = False

    @cached_class_property
    def openapi_types(cls):
        """
        This must be a method because a model may have properties that are
        of type self, this must run after the class is loaded

        Returns
            openapi_types (dict): The key is attribute name
                and the value is attribute type.
        """
        lazy_import()
        return {"value": ([ModelInfoMetric],)}

    @cached_class_property
    def discriminator(cls):
        return None

    attribute_map: Dict[str, str] = {}

    read_only_vars: Set[str] = set()

    _composed_schemas = None

    required_properties = set(
        [
            "_enforce_allowed_values",
            "_enforce_validations",
            "_data_store",
            "_check_type",
            "_spec_property_naming",
            "_path_to_item",
            "_configuration",
            "_visited_composed_classes",
        ]
    )

    @convert_js_args_to_python_args
    def __init__(self, *args, **kwargs) -> None:
        """ModelInfoSupportedMetrics - a model defined in OpenAPI

        Note that value can be passed either in args or in kwargs, but not in both.

        Args:
            args[0] ([ModelInfoMetric]): The distance metrics supported by the model for similarity search..  # noqa: E501

        Keyword Args:
            value ([ModelInfoMetric]): The distance metrics supported by the model for similarity search..  # noqa: E501
            _check_type (bool): if True, values for parameters in openapi_types
                                will be type checked and a TypeError will be
                                raised if the wrong type is input.
                                Defaults to True
            _path_to_item (tuple/list): This is a list of keys or values to
                                drill down to the model in received_data
                                when deserializing a response
            _spec_property_naming (bool): True if the variable names in the input data
                                are serialized names, as specified in the OpenAPI document.
                                False if the variable names in the input data
                                are pythonic names, e.g. snake case (default)
            _configuration (Configuration): the instance to use when
                                deserializing a file_type parameter.
                                If passed, type conversion is attempted
                                If omitted no type conversion is done.
            _visited_composed_classes (tuple): This stores a tuple of
                                classes that we have traveled through so that
                                if we see that class again we will not use its
                                discriminator again.
                                When traveling through a discriminator, the
                                composed schema that is
                                is traveled through is added to this set.
                                For example if Animal has a discriminator
                                petType and we pass in "Dog", and the class Dog
                                allOf includes Animal, we move through Animal
                                once using the discriminator, and pick Dog.
                                Then in Dog, we will make an instance of the
                                Animal class but this time we won't travel
                                through its discriminator because we passed in
                                _visited_composed_classes = (Animal,)
        """
        # required up here when default value is not given
        _path_to_item = kwargs.pop("_path_to_item", ())

        value = None
        if "value" in kwargs:
            value = kwargs.pop("value")

        if value is None and args:
            if len(args) == 1:
                value = args[0]
            elif len(args) > 1:
                raise PineconeApiTypeError(
                    "Invalid positional arguments=%s passed to %s. Remove those invalid positional arguments."
                    % (args, self.__class__.__name__),
                    path_to_item=_path_to_item,
                    valid_classes=(self.__class__,),
                )

        if value is None:
            raise PineconeApiTypeError(
                "value is required, but not passed in args or kwargs and doesn't have default",
                path_to_item=_path_to_item,
                valid_classes=(self.__class__,),
            )

        _enforce_allowed_values = kwargs.pop("_enforce_allowed_values", True)
        _enforce_validations = kwargs.pop("_enforce_validations", True)
        _check_type = kwargs.pop("_check_type", True)
        _spec_property_naming = kwargs.pop("_spec_property_naming", False)
        _configuration = kwargs.pop("_configuration", None)
        _visited_composed_classes = kwargs.pop("_visited_composed_classes", ())

        self._data_store = {}
        self._enforce_allowed_values = _enforce_allowed_values
        self._enforce_validations = _enforce_validations
        self._check_type = _check_type
        self._spec_property_naming = _spec_property_naming
        self._path_to_item = _path_to_item
        self._configuration = _configuration
        self._visited_composed_classes = _visited_composed_classes + (self.__class__,)
        self.value = value
        if kwargs:
            raise PineconeApiTypeError(
                "Invalid named arguments=%s passed to %s. Remove those invalid named arguments."
                % (kwargs, self.__class__.__name__),
                path_to_item=_path_to_item,
                valid_classes=(self.__class__,),
            )

    @classmethod
    @convert_js_args_to_python_args
    def _from_openapi_data(cls: Type[T], *args, **kwargs) -> T:
        """ModelInfoSupportedMetrics - a model defined in OpenAPI

        Note that value can be passed either in args or in kwargs, but not in both.

        Args:
            args[0] ([ModelInfoMetric]): The distance metrics supported by the model for similarity search.  # noqa: E501

        Keyword Args:
            value ([ModelInfoMetric]): The distance metrics supported by the model for similarity search.  # noqa: E501
            _check_type (bool): if True, values for parameters in openapi_types
                                will be type checked and a TypeError will be
                                raised if the wrong type is input.
                                Defaults to True
            _path_to_item (tuple/list): This is a list of keys or values to
                                drill down to the model in received_data
                                when deserializing a response
            _spec_property_naming (bool): True if the variable names in the input data
                                are serialized names, as specified in the OpenAPI document.
                                False if the variable names in the input data
                                are pythonic names, e.g. snake case (default)
            _configuration (Configuration): the instance to use when
                                deserializing a file_type parameter.
                                If passed, type conversion is attempted
                                If omitted no type conversion is done.
            _visited_composed_classes (tuple): This stores a tuple of
                                classes that we have traveled through so that
                                if we see that class again we will not use its
                                discriminator again.
                                When traveling through a discriminator, the
                                composed schema that is
                                is traveled through is added to this set.
                                For example if Animal has a discriminator
                                petType and we pass in "Dog", and the class Dog
                                allOf includes Animal, we move through Animal
                                once using the discriminator, and pick Dog.
                                Then in Dog, we will make an instance of the
                                Animal class but this time we won't travel
                                through its discriminator because we passed in
                                _visited_composed_classes = (Animal,)
        """
        # required up here when default value is not given
        _path_to_item = kwargs.pop("_path_to_item", ())

        self = super(OpenApiModel, cls).__new__(cls)

        value = None
        if "value" in kwargs:
            value = kwargs.pop("value")

        if value is None and args:
            if len(args) == 1:
                value = args[0]
            elif len(args) > 1:
                raise PineconeApiTypeError(
                    "Invalid positional arguments=%s passed to %s. Remove those invalid positional arguments."
                    % (args, self.__class__.__name__),
                    path_to_item=_path_to_item,
                    valid_classes=(self.__class__,),
                )

        if value is None:
            raise PineconeApiTypeError(
                "value is required, but not passed in args or kwargs and doesn't have default",
                path_to_item=_path_to_item,
                valid_classes=(self.__class__,),
            )

        _enforce_allowed_values = kwargs.pop("_enforce_allowed_values", False)
        _enforce_validations = kwargs.pop("_enforce_validations", False)
        _check_type = kwargs.pop("_check_type", True)
        _spec_property_naming = kwargs.pop("_spec_property_naming", False)
        _configuration = kwargs.pop("_configuration", None)
        _visited_composed_classes = kwargs.pop("_visited_composed_classes", ())

        self._data_store = {}
        self._enforce_allowed_values = _enforce_allowed_values
        self._enforce_validations = _enforce_validations
        self._check_type = _check_type
        self._spec_property_naming = _spec_property_naming
        self._path_to_item = _path_to_item
        self._configuration = _configuration
        self._visited_composed_classes = _visited_composed_classes + (self.__class__,)
        self.value = value
        if kwargs:
            raise PineconeApiTypeError(
                "Invalid named arguments=%s passed to %s. Remove those invalid named arguments."
                % (kwargs, self.__class__.__name__),
                path_to_item=_path_to_item,
                valid_classes=(self.__class__,),
            )

        return self
