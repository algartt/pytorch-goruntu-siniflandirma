import unittest

try:
    import torch
    from model import FashionCNN
except ImportError:
    torch = None
    FashionCNN = None


@unittest.skipIf(torch is None, "PyTorch kurulu değil")
class ModelTests(unittest.TestCase):
    def test_output_shape(self):
        model = FashionCNN()
        output = model(torch.randn(4, 1, 28, 28))
        self.assertEqual(tuple(output.shape), (4, 10))

    def test_model_has_trainable_parameters(self):
        model = FashionCNN()
        self.assertGreater(sum(parameter.numel() for parameter in model.parameters()), 0)


if __name__ == "__main__":
    unittest.main()
