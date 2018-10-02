import torch
import onnx
from .sdae import SDAE
import librosa


if __name__ == '__main__':
    model = onnx.load('test/speech_denoise.onnx')
    onnx.checker.check_model(model)
    graph = onnx.helper.printable_graph(model.graph)
    print(graph)

    model = torch.load('test/speech_denoise_model_e380.pth', map_location='cpu')
    print(model)
    dummy_input = torch.randn((10, 11 * 40))
    out = model(dummy_input)
    print(out.size())
    librosa.core.istft

