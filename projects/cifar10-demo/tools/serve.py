"""本地静态服务：电脑和手机都能访问，自动打印局域网地址和二维码。

用法：
    py tools/serve.py            # 默认 8000 端口，占用时自动往后找
    py tools/serve.py --port 8080
"""

import argparse
import ctypes
import json
import socket
import sys
import threading
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def local_ips():
    """收集本机可能被手机访问到的 IPv4 地址，把最可能的排在最前面。"""
    candidates = []
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.connect(("8.8.8.8", 80))
        candidates.append(probe.getsockname()[0])
        probe.close()
    except OSError:
        pass

    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            address = info[4][0]
            if address not in candidates:
                candidates.append(address)
    except OSError:
        pass

    return [
        address
        for address in candidates
        if not address.startswith("127.") and not address.startswith("169.254.")
    ]


def is_admin():
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def port_in_use(port):
    """真去连一下，比 bind 探测可靠：Windows 的 SO_REUSEADDR 会允许重复绑定。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(0.35)
        return probe.connect_ex(("127.0.0.1", port)) == 0


def pick_port(preferred):
    """从 preferred 开始找一个没人用的端口。"""
    for port in range(preferred, preferred + 20):
        if not port_in_use(port):
            return port
    raise SystemExit(f"{preferred} 到 {preferred + 19} 端口都被占用了，换个 --port 试试。")


class DemoHandler(SimpleHTTPRequestHandler):
    """模型和 WASM 体积大，允许缓存；HTML/JS/CSS 每次都校验，改完立刻生效。"""

    CACHEABLE = (".onnx", ".wasm", ".png")

    def end_headers(self):
        path = self.path.split("?")[0]
        if path.endswith(self.CACHEABLE):
            self.send_header("Cache-Control", "public, max-age=86400")
        else:
            self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, fmt, *args):
        status = args[1] if len(args) > 1 else ""
        sys.stdout.write(f"    {args[0]} -> {status}\n")


def print_qr(url):
    try:
        import qrcode
    except ImportError:
        print("    （没有安装 qrcode，想要扫码就用：py -m pip install qrcode）")
        return

    qr = qrcode.QRCode(border=1)
    qr.add_data(url)
    qr.make()
    qr.print_ascii(invert=True)


def write_web_assets(phone_url, port):
    """给页面准备两份东西：访问地址和一张白底二维码图。

    终端里的二维码对深色背景和字体很敏感，手机经常扫不出来；
    页面上这张是标准白底，扫起来稳定得多。
    """
    assets = ROOT / "assets"
    assets.mkdir(exist_ok=True)

    payload = {"phoneUrl": phone_url or "", "port": port}
    if phone_url:
        try:
            import qrcode
            import qrcode.image.svg

            image = qrcode.make(
                phone_url,
                image_factory=qrcode.image.svg.SvgPathImage,
                box_size=12,
                border=3,
            )
            image.save(str(assets / "qr.svg"))
        except Exception:
            (assets / "qr.svg").unlink(missing_ok=True)
            payload["phoneUrl"] = ""
    else:
        (assets / "qr.svg").unlink(missing_ok=True)

    (assets / "local.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true", help="不自动打开电脑浏览器")
    args = parser.parse_args()

    port = pick_port(args.port)
    addresses = local_ips()
    phone_url = f"http://{addresses[0]}:{port}" if addresses else ""
    write_web_assets(phone_url, port)

    print()
    print("  CIFAR-10 本地演示")
    print("  " + "-" * 46)
    print(f"  电脑访问   http://127.0.0.1:{port}")

    if addresses:
        print(f"  手机访问   {phone_url}")
        others = [a for a in addresses[1:] if not a.startswith("127.")][:3]
        for extra in others:
            print(f"  备用地址   http://{extra}:{port}")
        print()
        print("  手机和电脑连同一个 Wi-Fi，扫下面这个码：")
        print()
        print_qr(phone_url)
        print("  （页面上也有一张更清晰的二维码，扫不出来就用那个）")
    else:
        print("  手机访问   没检测到局域网地址，请检查是否连上了 Wi-Fi")

    print()
    if not is_admin():
        print("  连不上？多半是 Windows 防火墙拦住了。二选一：")
        print("    1. 弹出防火墙提示时选「允许访问」，勾上「专用网络」")
        print("    2. 用管理员身份运行一次下面这行，之后就不用再管：")
        print(f"       netsh advfirewall firewall add rule name=\"CIFAR10 Demo\" "
              f"dir=in action=allow protocol=TCP localport={port}")
        print()

    print("  按 Ctrl+C 停止服务")
    print("  " + "-" * 46)
    print()

    handler = partial(DemoHandler, directory=str(ROOT))
    server = ThreadingHTTPServer(("0.0.0.0", port), handler)

    if not args.no_open:
        threading.Timer(
            0.8, lambda: webbrowser.open(f"http://127.0.0.1:{port}")
        ).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务已停止。")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
