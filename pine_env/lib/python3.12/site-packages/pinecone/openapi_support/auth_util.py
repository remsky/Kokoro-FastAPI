from .exceptions import PineconeApiValueError


class AuthUtil:
    @staticmethod
    def update_params_for_auth(configuration, endpoint_auth_settings, headers, querys):
        """Updates header and query params based on authentication setting.

        :param headers: Header parameters dict to be updated.
        :param querys: Query parameters tuple list to be updated.
        :param auth_settings: Authentication setting identifiers list.
        """
        if not endpoint_auth_settings:
            return

        for auth in endpoint_auth_settings:
            auth_setting = configuration.auth_settings().get(auth)
            if auth_setting:
                if auth_setting["in"] == "header":
                    if auth_setting["type"] != "http-signature":
                        headers[auth_setting["key"]] = auth_setting["value"]
                elif auth_setting["in"] == "query":
                    querys.append((auth_setting["key"], auth_setting["value"]))
                else:
                    raise PineconeApiValueError(
                        "Authentication token must be in `query` or `header`"
                    )
