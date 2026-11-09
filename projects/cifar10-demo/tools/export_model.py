import argparse
import hashlib
import importlib.util
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


def load_model_definition():
    # 只导入模型定义，绝不导入会启动训练的脚本。
    project_dir = Path(__file__).resolve().parents[2]
    candidates = [
        project_dir / "resnet18" / "model_resnet18.py",
    ]
    for path in candidates:
        if path.is_file():
            spec = importlib.util.spec_from_file_location("cifar_resnet_model", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module.ResNet18()
    raise FileNotFoundError("找不到 ResNet 模型定义，请保留 projects/resnet18/model_resnet18.py")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--best-epoch", type=int, default=None)
    parser.add_argument("--validation-accuracy", type=float, default=None)
    parser.add_argument("--reported-test-accuracy", type=float, default=None)
    args = parser.parse_args()
    checkpoint_sha256 = hashlib.sha256(Path(args.weights).read_bytes()).hexdigest()

    output_dir = Path(args.output)
    samples_dir = output_dir / "assets" / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model_definition().to(device)
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

    model_sha256 = hashlib.sha256(onnx_path.read_bytes()).hexdigest()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    test_data = torchvision.datasets.CIFAR10(
        root=args.data_root, train=False, transform=transform, download=False
    )
    # 收集预测正确的样本，按该类正确样本置信度的 25% 分位选择展示图。
    # 这些是人工规则筛选的展示样本，不是随机抽样，也不能用来估计准确率。
    correct_by_class = [[] for _ in range(10)]
    total_correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for start in range(0, len(test_data), 256):
            images = torch.stack([
                test_data[index][0]
                for index in range(start, min(start + 256, len(test_data)))
            ]).to(device)
            labels = test_data.targets[start:start + images.shape[0]]
            logits = model(images)
            targets = torch.tensor(labels, device=device)
            total_loss += torch.nn.functional.cross_entropy(logits, targets, reduction="sum").item()
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            probabilities = torch.softmax(logits, dim=1).cpu()
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
        # 预测正确且置信度处于该分位的展示样本；不能代表总体难度。
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
            "file": f"assets/samples/{filename}?v=" + checkpoint_sha256[:12],
            "classIndex": class_index,
            "english": english,
            "chinese": chinese,
            "confidence": round(item["confidence"], 6),
            "sourceIndex": item["sourceIndex"],
        })
        print(f"{chinese}: test[{item['sourceIndex']}], confidence={item['confidence']:.4f}")

    metadata = {
        "modelName": "CIFAR ResNet18",
        "modelVersion": "resnet18-" + checkpoint_sha256[:12],
        "checkpointSha256": checkpoint_sha256,
        "modelSha256": model_sha256,
        "modelFile": "model.onnx?v=" + model_sha256[:12],
        "modelBytes": onnx_path.stat().st_size,
        "bestEpoch": args.best_epoch,
        "validationAccuracy": args.validation_accuracy,
        "reportedTestAccuracy": args.reported_test_accuracy,
        "exportTestAccuracy": total_correct / len(test_data),
        "exportTestLoss": total_loss / len(test_data),
        "sampleSelection": "correct predictions near per-class confidence 25th percentile; curated display only",
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
