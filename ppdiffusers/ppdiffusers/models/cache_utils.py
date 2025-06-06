from ..utils.logging import get_logger
logger = get_logger(__name__)


class CacheMixin:
    """
    A class for enable/disabling caching techniques on diffusion models.

    Supported caching techniques:
        - [Pyramid Attention Broadcast](https://huggingface.co/papers/2408.12588)
    """
    _cache_config = None

    @property
    def is_cache_enabled(self) ->bool:
        return self._cache_config is not None

    def enable_cache(self, config) ->None:
        """
        Enable caching techniques on the model.

        Args:
            config (`Union[PyramidAttentionBroadcastConfig]`):
                The configuration for applying the caching technique. Currently supported caching techniques are:
                    - [`~hooks.PyramidAttentionBroadcastConfig`]

        Example:

        ```python
        >>> import paddle
        >>> from diffusers import CogVideoXPipeline, PyramidAttentionBroadcastConfig

        >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-5b", paddle_dtype=paddle.bfloat16)

        >>> config = PyramidAttentionBroadcastConfig(
        ...     spatial_attention_block_skip_range=2,
        ...     spatial_attention_timestep_skip_range=(100, 800),
        ...     current_timestep_callback=lambda: pipe.current_timestep,
        ... )
        >>> pipe.transformer.enable_cache(config)
        ```
        """
        from ..hooks import PyramidAttentionBroadcastConfig, apply_pyramid_attention_broadcast
        if isinstance(config, PyramidAttentionBroadcastConfig):
            apply_pyramid_attention_broadcast(self, config)
        else:
            raise ValueError(f'Cache config {type(config)} is not supported.')
        self._cache_config = config

    def disable_cache(self) ->None:
        from ..hooks import HookRegistry, PyramidAttentionBroadcastConfig
        if self._cache_config is None:
            logger.warning(
                "Caching techniques have not been enabled, so there's nothing to disable."
                )
            return
        if isinstance(self._cache_config, PyramidAttentionBroadcastConfig):
            registry = HookRegistry.check_if_exists_or_initialize(self)
            registry.remove_hook('pyramid_attention_broadcast', recurse=True)
        else:
            raise ValueError(
                f'Cache config {type(self._cache_config)} is not supported.')
        self._cache_config = None

    def _reset_stateful_cache(self, recurse: bool=True) ->None:
        from ..hooks import HookRegistry
        HookRegistry.check_if_exists_or_initialize(self).reset_stateful_hooks(
            recurse=recurse)
