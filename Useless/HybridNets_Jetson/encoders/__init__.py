from .efficientnet import efficient_net_encoders


encoders = {}
encoders.update(efficient_net_encoders)


def get_encoder(name, in_channels=3, depth=5, weights=None, output_stride=32, **kwargs):
    if output_stride != 32:
        raise ValueError(f"Runtime bundle only supports output_stride=32, got {output_stride}.")

    try:
        encoder_cls = encoders[name]["encoder"]
    except KeyError as exc:
        raise KeyError(f"Unsupported encoder `{name}` in runtime bundle.") from exc

    params = dict(encoders[name]["params"])
    params.update(depth=depth)
    encoder = encoder_cls(**params)

    if weights is not None:
        raise RuntimeError(
            "Pretrained encoder downloads are disabled in the Jetson runtime bundle. "
            "Load the packaged checkpoint instead."
        )

    encoder.set_in_channels(in_channels, pretrained=False)
    return encoder


def get_encoder_names():
    return list(encoders.keys())
