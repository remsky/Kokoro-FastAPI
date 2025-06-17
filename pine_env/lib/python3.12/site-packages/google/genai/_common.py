# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Common utilities for the SDK."""

import base64
import datetime
import enum
import functools
import typing
from typing import Any, Callable, Optional, Union, get_origin, get_args
import uuid
import warnings

import pydantic
from pydantic import alias_generators

from . import _api_client
from . import errors


def set_value_by_path(data: Optional[dict[Any, Any]], keys: list[str], value: Any) -> None:
  """Examples:

  set_value_by_path({}, ['a', 'b'], v)
    -> {'a': {'b': v}}
  set_value_by_path({}, ['a', 'b[]', c], [v1, v2])
    -> {'a': {'b': [{'c': v1}, {'c': v2}]}}
  set_value_by_path({'a': {'b': [{'c': v1}, {'c': v2}]}}, ['a', 'b[]', 'd'], v3)
    -> {'a': {'b': [{'c': v1, 'd': v3}, {'c': v2, 'd': v3}]}}
  """
  if value is None:
    return
  for i, key in enumerate(keys[:-1]):
    if key.endswith('[]'):
      key_name = key[:-2]
      if data is not None and key_name not in data:
        if isinstance(value, list):
          data[key_name] = [{} for _ in range(len(value))]
        else:
          raise ValueError(
              f'value {value} must be a list given an array path {key}'
          )
      if isinstance(value, list) and data is not None:
        for j, d in enumerate(data[key_name]):
          set_value_by_path(d, keys[i + 1 :], value[j])
      else:
        if data is not None:
          for d in data[key_name]:
            set_value_by_path(d, keys[i + 1 :], value)
      return
    elif key.endswith('[0]'):
      key_name = key[:-3]
      if data is not None and key_name not in data:
        data[key_name] = [{}]
      if data is not None:
        set_value_by_path(data[key_name][0], keys[i + 1 :], value)
      return
    if data is not None:
      data = data.setdefault(key, {})

  if data is not None:
    existing_data = data.get(keys[-1])
    # If there is an existing value, merge, not overwrite.
    if existing_data is not None:
      # Don't overwrite existing non-empty value with new empty value.
      # This is triggered when handling tuning datasets.
      if not value:
        pass
      # Don't fail when overwriting value with same value
      elif value == existing_data:
        pass
      # Instead of overwriting dictionary with another dictionary, merge them.
      # This is important for handling training and validation datasets in tuning.
      elif isinstance(existing_data, dict) and isinstance(value, dict):
        # Merging dictionaries. Consider deep merging in the future.
        existing_data.update(value)
      else:
        raise ValueError(
            f'Cannot set value for an existing key. Key: {keys[-1]};'
            f' Existing value: {existing_data}; New value: {value}.'
        )
    else:
      data[keys[-1]] = value


def get_value_by_path(data: Any, keys: list[str]) -> Any:
  """Examples:

  get_value_by_path({'a': {'b': v}}, ['a', 'b'])
    -> v
  get_value_by_path({'a': {'b': [{'c': v1}, {'c': v2}]}}, ['a', 'b[]', 'c'])
    -> [v1, v2]
  """
  if keys == ['_self']:
    return data
  for i, key in enumerate(keys):
    if not data:
      return None
    if key.endswith('[]'):
      key_name = key[:-2]
      if key_name in data:
        return [get_value_by_path(d, keys[i + 1 :]) for d in data[key_name]]
      else:
        return None
    elif key.endswith('[0]'):
      key_name = key[:-3]
      if key_name in data and data[key_name]:
        return get_value_by_path(data[key_name][0], keys[i + 1 :])
      else:
        return None
    else:
      if key in data:
        data = data[key]
      elif isinstance(data, BaseModel) and hasattr(data, key):
        data = getattr(data, key)
      else:
        return None
  return data


def convert_to_dict(obj: object) -> Any:
  """Recursively converts a given object to a dictionary.

  If the object is a Pydantic model, it uses the model's `model_dump()` method.

  Args:
    obj: The object to convert.

  Returns:
    A dictionary representation of the object, a list of objects if a list is
    passed, or the object itself if it is not a dictionary, list, or Pydantic
    model.
  """
  if isinstance(obj, pydantic.BaseModel):
    return obj.model_dump(exclude_none=True)
  elif isinstance(obj, dict):
    return {key: convert_to_dict(value) for key, value in obj.items()}
  elif isinstance(obj, list):
    return [convert_to_dict(item) for item in obj]
  else:
    return obj


def _is_struct_type(annotation: type) -> bool:
  """Checks if the given annotation is list[dict[str, typing.Any]]
  or typing.List[typing.Dict[str, typing.Any]].

  This maps to Struct type in the API.
  """
  outer_origin = get_origin(annotation)
  outer_args = get_args(annotation)

  if outer_origin is not list: # Python 3.9+ normalizes list
    return False

  if not outer_args or len(outer_args) != 1:
    return False

  inner_annotation = outer_args[0]

  inner_origin = get_origin(inner_annotation)
  inner_args = get_args(inner_annotation)

  if inner_origin is not dict: # Python 3.9+ normalizes to dict
    return False

  if not inner_args or len(inner_args) != 2:
    # dict should have exactly two type arguments
    return False

  # Check if the dict arguments are str and typing.Any
  key_type, value_type = inner_args
  return key_type is str and value_type is typing.Any


def _remove_extra_fields(
    model: Any, response: dict[str, object]
) -> None:
  """Removes extra fields from the response that are not in the model.

  Mutates the response in place.
  """

  key_values = list(response.items())

  for key, value in key_values:
    # Need to convert to snake case to match model fields names
    # ex: UsageMetadata
    alias_map = {
        field_info.alias: key for key, field_info in model.model_fields.items()
    }

    if key not in model.model_fields and key not in alias_map:
      response.pop(key)
      continue

    key = alias_map.get(key, key)

    annotation = model.model_fields[key].annotation

    # Get the BaseModel if Optional
    if typing.get_origin(annotation) is Union:
      annotation = typing.get_args(annotation)[0]

    # if dict, assume BaseModel but also check that field type is not dict
    # example: FunctionCall.args
    if isinstance(value, dict) and typing.get_origin(annotation) is not dict:
      _remove_extra_fields(annotation, value)
    elif isinstance(value, list):
      if _is_struct_type(annotation):
        continue

      for item in value:
        # assume a list of dict is list of BaseModel
        if isinstance(item, dict):
          _remove_extra_fields(typing.get_args(annotation)[0], item)

T = typing.TypeVar('T', bound='BaseModel')


class BaseModel(pydantic.BaseModel):

  model_config = pydantic.ConfigDict(
      alias_generator=alias_generators.to_camel,
      populate_by_name=True,
      from_attributes=True,
      protected_namespaces=(),
      extra='forbid',
      # This allows us to use arbitrary types in the model. E.g. PIL.Image.
      arbitrary_types_allowed=True,
      ser_json_bytes='base64',
      val_json_bytes='base64',
      ignored_types=(typing.TypeVar,)
  )

  @classmethod
  def _from_response(
      cls: typing.Type[T], *, response: dict[str, object], kwargs: dict[str, object]
  ) -> T:
    # To maintain forward compatibility, we need to remove extra fields from
    # the response.
    # We will provide another mechanism to allow users to access these fields.

    # For Agent Engine we don't want to call _remove_all_fields because the
    # user may pass a dict that is not a subclass of BaseModel.
    # If more modules require we skip this, we may want a different approach
    should_skip_removing_fields = (
        kwargs is not None and
        'config' in kwargs and
        kwargs['config'] is not None and
        isinstance(kwargs['config'], dict) and
        'include_all_fields' in kwargs['config']
        and kwargs['config']['include_all_fields']
    )

    if not should_skip_removing_fields:
      _remove_extra_fields(cls, response)
    validated_response = cls.model_validate(response)
    return validated_response

  def to_json_dict(self) -> dict[str, object]:
    return self.model_dump(exclude_none=True, mode='json')


class CaseInSensitiveEnum(str, enum.Enum):
  """Case insensitive enum."""

  @classmethod
  def _missing_(cls, value: Any) -> Any:
    try:
      return cls[value.upper()]  # Try to access directly with uppercase
    except KeyError:
      try:
        return cls[value.lower()]  # Try to access directly with lowercase
      except KeyError:
        warnings.warn(f"{value} is not a valid {cls.__name__}")
        try:
          # Creating a enum instance based on the value
          # We need to use super() to avoid infinite recursion.
          unknown_enum_val = super().__new__(cls, value)
          unknown_enum_val._name_ = str(value)  # pylint: disable=protected-access
          unknown_enum_val._value_ = value  # pylint: disable=protected-access
          return unknown_enum_val
        except:
          return None


def timestamped_unique_name() -> str:
  """Composes a timestamped unique name.

  Returns:
      A string representing a unique name.
  """
  timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
  unique_id = uuid.uuid4().hex[0:5]
  return f'{timestamp}_{unique_id}'


def encode_unserializable_types(data: dict[str, object]) -> dict[str, object]:
  """Converts unserializable types in dict to json.dumps() compatible types.

  This function is called in models.py after calling convert_to_dict(). The
  convert_to_dict() can convert pydantic object to dict. However, the input to
  convert_to_dict() is dict mixed of pydantic object and nested dict(the output
  of converters). So they may be bytes in the dict and they are out of
  `ser_json_bytes` control in model_dump(mode='json') called in
  `convert_to_dict`, as well as datetime deserialization in Pydantic json mode.

  Returns:
    A dictionary with json.dumps() incompatible type (e.g. bytes datetime)
    to compatible type (e.g. base64 encoded string, isoformat date string).
  """
  processed_data: dict[str, object] = {}
  if not isinstance(data, dict):
    return data
  for key, value in data.items():
    if isinstance(value, bytes):
      processed_data[key] = base64.urlsafe_b64encode(value).decode('ascii')
    elif isinstance(value, datetime.datetime):
      processed_data[key] = value.isoformat()
    elif isinstance(value, dict):
      processed_data[key] = encode_unserializable_types(value)
    elif isinstance(value, list):
      if all(isinstance(v, bytes) for v in value):
        processed_data[key] = [
            base64.urlsafe_b64encode(v).decode('ascii') for v in value
        ]
      if all(isinstance(v, datetime.datetime) for v in value):
        processed_data[key] = [v.isoformat() for v in value]
      else:
        processed_data[key] = [encode_unserializable_types(v) for v in value]
    else:
      processed_data[key] = value
  return processed_data


def experimental_warning(message: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
  """Experimental warning, only warns once."""
  def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
    warning_done = False
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
      nonlocal warning_done
      if not warning_done:
        warning_done = True
        warnings.warn(
            message=message,
            category=errors.ExperimentalWarning,
            stacklevel=2,
        )
      return func(*args, **kwargs)
    return wrapper
  return decorator

