import argparse
import json
from pathlib import Path

import torch
import torchvision
from PIL import Image
from torchvision import transforms


CLASSES = [
    ("airplane", "飞机"),
    ("automobile", "汽车"),
    ("bird", "鸟"),
    ("cat", "猫"),
    ("deer", "鹿"),
    ("dog", "狗"),
    ("frog", "青蛙"),
    ("horse", "马"),
    ("ship", "船"),
    ("truck", "卡车"),
]


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = torch.nn.Conv2d(16, 32, 3, 1, 1)
        self.conv3 = torch.nn.Conv2d(32, 64, 3, 1, 1)
        self.conv4 = torch.nn.Conv2d(64, 128, 3, 1, 1)
        self.conv5 = torch.nn.Conv2d(128, 256, 3, 1, 1)
        self.conv6 = torch.nn.Conv2d(256, 512, 3, 1, 1)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.flatten = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(512 * 4 * 4, 512)
        self.fc2 = torch.nn.Linear(512, 64)
        self.fc3 = torch.nn.Linear(64, 10)
        self.bn1 = torch.nn.BatchNorm2d(16)
        self.bn2 = torch.nn.BatchNorm2d(32)
        self.bn3 = torch.nn.BatchNorm2d(64)
        self.bn4 = torch.nn.BatchNorm2d(128)
        self.bn5 = torch.nn.BatchNorm2d(256)
        self.bn6 = torch.nn.BatchNorm2d(512)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.pool(self.relu(self.bn2(self.conv2(
            self.relu(self.bn1(self.conv1(x)))
        ))))
        x = self.pool(self.relu(self.bn4(self.conv4(
            self.relu(self.bn3(self.conv3(x)))
        ))))
        x = self.pool(self.relu(self.bn6(self.conv6(
            self.relu(self.bn5(self.conv5(x)))
        ))))
        x = self.flatten(x)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output)
    samples_dir = output_dir / "assets" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = Net().to(device)
    state_dict = torch.load(args.weights, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    dummy = torch.zeros(1, 3, 32, 32, device=device)
    onnx_path = output_dir / "model.onnx"
    torch.onnx.export(
        model,
        dummy,
        onnx_path,
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    test_data = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, transform=transform, download=False
    )
    # 收集每一类所有预测正确的样本，稍后挑选“接近该类中位数置信度”的那张，
    # 避免清一色 100% 的极端样本，让示例更能代表模型的真实表现。
    correct_by_class = [[] for _ in range(10)]

    with torch.no_grad():
        for start in range(0, len(test_data), 256):
            images = torch.stack([
                test_data[index][0]
                for index in range(start, min(start + 256, len(test_data)))
            ]).to(device)
            labels = test_data.targets[start:start + images.shape[0]]
            probabilities = torch.softmax(model(images), dim=1).cpu()
            predictions = probabilities.argmax(dim=1).tolist()
            for offset, (label, prediction) in enumerate(zip(labels, predictions)):
                if prediction != label:
                    continue
                correct_by_class[label].append(
                    (start + offset, probabilities[offset, label].item())
                )

    best = [None] * 10
    for class_index, items in enumerate(correct_by_class):
        if not items:
            raise RuntimeError(f"没有找到类别 {class_index} 的正确示例")
        # 用 25% 分位的置信度作为目标：模型整体很自信，取低分位才能选到
        # 更能反映真实难度、又确实预测正确的样本。
        confidences = sorted(confidence for _, confidence in items)
        target = confidences[int(len(confidences) * 0.25)]
        index, confidence = min(items, key=lambda item: abs(item[1] - target))
        best[class_index] = {"sourceIndex": index, "confidence": confidence}

    samples = []
    for class_index, item in enumerate(best):
        if item is None:
            raise RuntimeError(f"没有找到类别 {class_index} 的正确示例")
        english, chinese = CLASSES[class_index]
        filename = f"{class_index}-{english}.png"
        image = Image.fromarray(test_data.data[item["sourceIndex"]])
        image.save(samples_dir / filename)
        samples.append({
            "file": f"assets/samples/{filename}",
            "classIndex": class_index,
            "english": english,
            "chinese": chinese,
            "confidence": round(item["confidence"], 6),
            "sourceIndex": item["sourceIndex"],
        })
        print(f"{chinese}: test[{item['sourceIndex']}], confidence={item['confidence']:.4f}")

    metadata = {
        "inputShape": [1, 3, 32, 32],
        "normalization": "(pixel / 255 - 0.5) / 0.5",
        "modelParameters": sum(parameter.numel() for parameter in model.parameters()),
        "samples": samples,
    }
    (output_dir / "assets" / "samples.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"ONNX: {onnx_path} ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")


if __name__ == "__main__":
    main()
