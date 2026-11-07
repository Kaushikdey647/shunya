# HTTP settings and errors

`Settings` reads `SHUNYA_API_*` (and the repo `.env` when present). Domain error codes are defined in `shunya.errors` and re-exported here. FastAPI maps `ShunyaError` in `api.exception_handlers`.

Field-level environment documentation stays in [HTTP API overview](http-api-package.md) and [`api/README.md`](https://github.com/Kaushikdey647/shunya/blob/main/api/README.md).

## Settings

::: api.settings
    options:
      show_submodules: false
      heading_level: 3
      members_order: source
      filters:
        - "!^_"

## API error re-exports

::: api.errors
    options:
      show_submodules: false
      heading_level: 3
      members_order: source
      filters:
        - "!^_"
