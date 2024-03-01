import hashlib
import os
import time
import urllib
import warnings
from functools import partial
from typing import Dict, Union

from tqdm import tqdm

from .version import __version__

try:
    from huggingface_hub import hf_hub_download
    hf_hub_download = partial(
        hf_hub_download, library_name="open_clip", library_version=__version__)
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False


def _pcfg(url='', hf_hub='', mean=None, std=None):
    return dict(
        url=url,
        hf_hub=hf_hub,
        mean=mean,
        std=std,
    )


_RN50 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt"),
    cc12m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt"),
)

_RN50_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-yfcc15m-455df137.pt"),
    cc12m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn50-quickgelu-cc12m-f000538c.pt"),
)

_RN101 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt"),
)

_RN101_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/8fa8567bab74a42d41c5915025a8e4538c3bdbe8804a470a72f30b0d94fab599/RN101.pt"),
    yfcc15m=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/rn101-quickgelu-yfcc15m-3e04b30e.pt"),
)

_RN50x4 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/7e526bd135e493cef0776de27d5f42653e6b4c8bf9e0f653bb11773263205fdd/RN50x4.pt"),
)

_RN50x16 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/52378b407f34354e150460fe41077663dd5b39c54cd0bfd2b27167a4a06ec9aa/RN50x16.pt"),
)

_RN50x64 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/be1cfb55d75a9666199fb2206c106743da0f6468c9d327f3e0d0a543a9919d9c/RN50x64.pt"),
)

_VITB32 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt"),
    laion2b_e16=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-laion2b_e16-af8dbd0c.pth"),
    laion2b_s34b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-B-32-laion2B-s34B-b79K/')
)

_VITB32_quickgelu = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e31-d867053b.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_32-quickgelu-laion400m_e32-46683a32.pt"),
)

_VITB16 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e31-00efa78f.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16-laion400m_e32-55e67d44.pt"),
    # laion400m_32k=_pcfg(
    #     url="",
    #     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    # laion400m_64k=_pcfg(
    #     url="",
    #     mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    laion2b_s34b_b88k=_pcfg(hf_hub='laion/CLIP-ViT-B-16-laion2B-s34B-b88K/'),
)

_VITB16_PLUS_240 = dict(
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e32-699c4b84.pt"),
)

_VITL14 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt"),
    laion400m_e31=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e31-69988bb6.pt"),
    laion400m_e32=_pcfg(
        "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_l_14-laion400m_e32-3d133497.pt"),
    laion2b_s32b_b82k=_pcfg(
        hf_hub='laion/CLIP-ViT-L-14-laion2B-s32B-b82K/',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
)

_VITL14_336 = dict(
    openai=_pcfg(
        "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt"),
)

_VITH14 = dict(
    laion2b_s32b_b79k=_pcfg(hf_hub='laion/CLIP-ViT-H-14-laion2B-s32B-b79K/'),
)

_VITg14 = dict(
    laion2b_s12b_b42k=_pcfg(hf_hub='laion/CLIP-ViT-g-14-laion2B-s12B-b42K/'),
)

# TinyCLIP

# manual weight inheritance

_TINYCLIP_VIT_39M_16_TEXT_19M = {
    "YFCC15M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-ViT-39M-16-Text-19M-YFCC15M.pt",
    ),
}

_TINYCLIP_VIT_8M_16_TEXT_3M = {
    "YFCC15M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M.pt",
    ),
}

_TINYCLIP_RESNET_30M_TEXT_29M = {
    "LAION400M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-ResNet-30M-Text-29M-LAION400M.pt",
    ),
}

_TINYCLIP_RESNET_19M_TEXT_19M = {
    "LAION400M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-ResNet-19M-Text-19M-LAION400M.pt",
    ),
}

_TINYCLIP_VIT_61M_32_TEXT_29M = {
    "LAION400M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-ViT-61M-32-Text-29M-LAION400M.pt",
    ),
}

_TINYCLIP_VIT_40M_32_TEXT_19M = {
    "LAION400M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-ViT-40M-32-Text-19M-LAION400M.pt",
    ),
}

# auto weight inheritance

_TINYCLIP_AUTO_VIT_63M_32_TEXT_31M = {
    "LAION400M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-auto-ViT-63M-32-Text-31M-LAION400M.pt",
    ),
    "LAIONYFCC400M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-auto-ViT-63M-32-Text-31M-LAIONYFCC400M.pt",
    ),
}

_TINYCLIP_AUTO_VIT_45M_32_TEXT_18M = {
    "LAION400M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-auto-ViT-45M-32-Text-18M-LAION400M.pt",
    ),
    "LAIONYFCC400M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-auto-ViT-45M-32-Text-18M-LAIONYFCC400M.pt",
    ),
}

_TINYCLIP_AUTO_VIT_22M_32_TEXT_10M = {
    "LAION400M": _pcfg(
        "https://github.com/wkcn/TinyCLIP-model-zoo/releases/download/checkpoints/TinyCLIP-auto-ViT-22M-32-Text-10M-LAION400M.pt",
    ),
}

_PRETRAINED = {
    "RN50": _RN50,
    "RN50-quickgelu": _RN50_quickgelu,
    "RN101": _RN101,
    "RN101-quickgelu": _RN101_quickgelu,
    "RN50x4": _RN50x4,
    "RN50x16": _RN50x16,
    "RN50x64": _RN50x64,
    "ViT-B-32": _VITB32,
    "ViT-B-32-quickgelu": _VITB32_quickgelu,
    "ViT-B-16": _VITB16,
    "ViT-B-16-plus-240": _VITB16_PLUS_240,
    "ViT-L-14": _VITL14,
    "ViT-L-14-336": _VITL14_336,
    "ViT-H-14": _VITH14,
    "ViT-g-14": _VITg14,

    "TinyCLIP-ViT-39M-16-Text-19M": _TINYCLIP_VIT_39M_16_TEXT_19M,
    "TinyCLIP-ViT-8M-16-Text-3M": _TINYCLIP_VIT_8M_16_TEXT_3M,
    "TinyCLIP-ResNet-30M-Text-29M": _TINYCLIP_RESNET_30M_TEXT_29M,
    "TinyCLIP-ResNet-19M-Text-19M": _TINYCLIP_RESNET_19M_TEXT_19M,
    "TinyCLIP-ViT-61M-32-Text-29M": _TINYCLIP_VIT_61M_32_TEXT_29M,
    "TinyCLIP-ViT-40M-32-Text-19M": _TINYCLIP_VIT_40M_32_TEXT_19M,

    "TinyCLIP-auto-ViT-63M-32-Text-31M": _TINYCLIP_AUTO_VIT_63M_32_TEXT_31M,
    "TinyCLIP-auto-ViT-45M-32-Text-18M": _TINYCLIP_AUTO_VIT_45M_32_TEXT_18M,
    "TinyCLIP-auto-ViT-22M-32-Text-10M": _TINYCLIP_AUTO_VIT_22M_32_TEXT_10M,
}


def list_pretrained(as_str: bool = False):
    """ returns list of pretrained models
    Returns a tuple (model_name, pretrain_tag) by default or 'name:tag' if as_str == True
    """
    return [':'.join([k, t]) if as_str else (k, t) for k in _PRETRAINED.keys() for t in _PRETRAINED[k].keys()]


def list_pretrained_tag_models(tag: str):
    """ return all models having the specified pretrain tag """
    models = []
    for k in _PRETRAINED.keys():
        if tag in _PRETRAINED[k]:
            models.append(k)
    return models


def list_pretrained_model_tags(model: str):
    """ return all pretrain tags for the specified model architecture """
    tags = []
    if model in _PRETRAINED:
        tags.extend(_PRETRAINED[model].keys())
    return tags


def is_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return False
    return tag.lower() in _PRETRAINED[model]


def get_pretrained_cfg(model: str, tag: str):
    if model not in _PRETRAINED:
        return {}
    model_pretrained = _PRETRAINED[model]
    if tag in model_pretrained:
        return model_pretrained[tag]
    return model_pretrained.get(tag.lower(), {})


def get_pretrained_url(model: str, tag: str):
    cfg = get_pretrained_cfg(model, tag)
    return cfg.get('url', '')


def is_local_master():
    return int(os.getenv('LOCAL_RANK', 0)) == 0


def download_pretrained_from_url(
        url: str = os.path.expanduser("~/.cache/clip"),
        cache_dir: Union[str, None] = None,
):

    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)

    filename = os.path.basename(url)
    download_target = os.path.join(cache_dir, filename)
    if is_local_master():
        for _ in range(20):
            try:
                return _download_pretrained(url, cache_dir)
            except Exception as e:
                print(f'Download pretrained: {url}, {cache_dir}, {e}')
                time.sleep(10)
    else:
        while not os.path.exists(download_target):
            time.sleep(1)
    return download_target


def _download_pretrained(url: str, root: str = os.path.expanduser("~/.cache/clip")):
    os.makedirs(root, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    else:
        expected_sha256 = ''

    download_target = os.path.join(root, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(
            f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest() == expected_sha256:
                return download_target
            else:
                warnings.warn(
                    f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    download_target_tmp = download_target + ".tmp"
    with urllib.request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and hashlib.sha256(open(download_target_tmp, "rb").read()).hexdigest() != expected_sha256:
        os.remove(download_target_tmp)
        raise RuntimeError(
            f"Model has been downloaded but the SHA256 checksum does not not match")

    os.rename(download_target_tmp, download_target)
    return download_target


def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub


def download_pretrained_from_hf(
        model_id: str,
        filename: str = 'open_clip_pytorch_model.bin',
        revision=None,
        cache_dir: Union[str, None] = None,
):
    has_hf_hub(True)
    cached_file = hf_hub_download(
        model_id, filename, revision=revision, cache_dir=cache_dir)
    return cached_file


def download_pretrained(
        cfg: Dict,
        force_hf_hub: bool = False,
        cache_dir: Union[str, None] = None,
):
    target = ''
    if not cfg:
        return target

    download_url = cfg.get('url', '')
    download_hf_hub = cfg.get('hf_hub', '')
    if download_hf_hub and force_hf_hub:
        # use HF hub even if url exists
        download_url = ''

    if download_url:
        target = download_pretrained_from_url(
            download_url, cache_dir=cache_dir)
    elif download_hf_hub:
        has_hf_hub(True)
        # we assume the hf_hub entries in pretrained config combine model_id + filename in
        # 'org/model_name/filename.pt' form. To specify just the model id w/o filename and
        # use 'open_clip_pytorch_model.bin' default, there must be a trailing slash 'org/model_name/'.
        model_id, filename = os.path.split(download_hf_hub)
        if filename:
            target = download_pretrained_from_hf(
                model_id, filename=filename, cache_dir=cache_dir)
        else:
            target = download_pretrained_from_hf(model_id, cache_dir=cache_dir)

    return target
