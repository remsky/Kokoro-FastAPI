class cached_class_property:
    def __init__(self, func) -> None:
        self.func = func
        self.attr_name = f"__cached_{func.__name__}"

    def __get__(self, instance, owner):
        # The value is stored on the owner (the class), not the instance.
        if hasattr(owner, self.attr_name):
            return getattr(owner, self.attr_name)
        value = self.func(owner)
        setattr(owner, self.attr_name, value)
        return value
