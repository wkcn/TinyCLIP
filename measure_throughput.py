import os
import sys
import open_clip
import time
import torch
import argparse

cuda_is_available = torch.cuda.is_available()


@torch.no_grad()
@torch.cuda.amp.autocast()
def benchmark(fn, batch_size):
    WARMUP_T = 10
    T = 50
    for _ in range(WARMUP_T):
        fn()
    if cuda_is_available:
        torch.cuda.synchronize()
    tic1 = time.time()
    for _ in range(T):
        fn()
        if cuda_is_available:
            torch.cuda.synchronize()
    tic2 = time.time()
    throughput = batch_size * T / (tic2 - tic1)
    return throughput


def throughput(model_name, batch_size, image_size, device):
    model, _, _ = open_clip.create_model_and_transforms(model_name)
    model = model.to(device)
    model.eval()

    image = torch.rand((batch_size, 3, image_size, image_size), device=device)
    text = open_clip.tokenize(["a photo of a cat"]).to(device)
    text = text.repeat(batch_size, 1)

    def image_fn():
        model.encode_image(image)
    def text_fn():
        model.encode_text(text)
    image_throughput = benchmark(image_fn, batch_size)
    text_throughput = benchmark(text_fn, batch_size)
    throughput = 1.0 / (1.0 / image_throughput + 1.0 / text_throughput)
    print(f'Image throughput: {image_throughput} images/sec')
    print(f'Text throughput: {text_throughput} texts/sec')
    print(f'Pair throughput: {throughput} pairs/sec')


def parse_option():
    parser = argparse.ArgumentParser('measure throughput')
    parser.add_argument('--model-name', type=str, default='ViT-B/32',
                        help='model name')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--image-size', type=int, default=224,
                        help='image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='device')
    opt = parser.parse_args()
    return opt


if __name__ == '__main__':
    opt = parse_option()
    throughput(opt.model_name, opt.batch_size, opt.image_size, opt.device)
