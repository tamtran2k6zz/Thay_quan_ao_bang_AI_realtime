"""
HTTPS Local Server cho iPhone Safari (Web App Virtual Try-On).
iOS Safari bắt buộc HTTPS mới cho phép mở Camera qua IP mạng LAN (192.168.x.x).

Chạy từ thư mục gốc dự án:
    python web/serve_https.py
Sau đó mở trên Safari iPhone:
    https://<IP-MÁY-WINDOWS>:8443
"""

import argparse
import datetime
import ipaddress
import json
import mimetypes
import socket
import ssl
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID

WEB_DIR = Path(__file__).resolve().parent
ROOT_DIR = WEB_DIR.parent
DATASET_DIR = ROOT_DIR / "clothes_dataset"
CERT_DIR = WEB_DIR / ".certs"

mimetypes.add_type("application/javascript", ".js")
mimetypes.add_type("application/wasm", ".wasm")
mimetypes.add_type("application/octet-stream", ".onnx")
mimetypes.add_type("application/manifest+json", ".webmanifest")


def detect_lan_ip() -> str:
    """Tìm địa chỉ IPv4 mạng LAN của máy Windows hiện tại."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return sock.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        sock.close()


def ensure_self_signed_cert(lan_ip: str) -> tuple[Path, Path]:
    """Tạo (hoặc cập nhật nếu đổi IP LAN) chứng chỉ SSL tự ký có SAN chuẩn cho Safari iOS."""
    CERT_DIR.mkdir(parents=True, exist_ok=True)
    cert_path = CERT_DIR / "cert.pem"
    key_path = CERT_DIR / "key.pem"
    ip_marker = CERT_DIR / "lan_ip.txt"

    if cert_path.exists() and key_path.exists() and ip_marker.exists():
        if ip_marker.read_text(encoding="utf-8").strip() == lan_ip:
            return cert_path, key_path

    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name(
        [
            x509.NameAttribute(NameOID.COUNTRY_NAME, "VN"),
            x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Rust Try-On Local"),
            x509.NameAttribute(NameOID.COMMON_NAME, lan_ip),
        ]
    )

    san_entries: list[x509.GeneralName] = [
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.IPv4Address("127.0.0.1")),
    ]
    try:
        parsed_ip = ipaddress.ip_address(lan_ip)
        if str(parsed_ip) != "127.0.0.1":
            san_entries.append(x509.IPAddress(parsed_ip))
    except ValueError:
        pass

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(private_key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now - datetime.timedelta(minutes=5))
        .not_valid_after(now + datetime.timedelta(days=365))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=False, path_length=None), critical=True)
        .add_extension(
            x509.KeyUsage(
                digital_signature=True,
                key_encipherment=True,
                content_commitment=False,
                data_encipherment=False,
                key_agreement=False,
                key_cert_sign=False,
                crl_sign=False,
                encipher_only=False,
                decipher_only=False,
            ),
            critical=True,
        )
        .add_extension(
            x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]),
            critical=False,
        )
        .sign(private_key, hashes.SHA256())
    )

    key_path.write_bytes(
        private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        )
    )
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    ip_marker.write_text(lan_ip, encoding="utf-8")
    return cert_path, key_path


class TryOnRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format: str, *args) -> None:
        print(f"[HTTPS] {self.address_string()} - {format % args}")

    def do_GET(self) -> None:
        req_path = unquote(self.path.split("?", 1)[0])

        if req_path == "/api/dataset":
            self._serve_dataset_manifest()
            return

        if req_path == "/":
            req_path = "/index.html"

        # Phục vụ ảnh dataset & ảnh tách nền từ `clothes_dataset/{images,garments_rgba}/...`
        if req_path.startswith("/images/") or req_path.startswith("/garments_rgba/"):
            target = (DATASET_DIR / req_path.lstrip("/")).resolve()
            if self._is_within(target, DATASET_DIR) and target.is_file():
                self._serve_file(target)
                return
            self.send_error(404, "Dataset asset not found")
            return

        # Phục vụ file tĩnh trong `web/` (tự ánh xạ .onnx từ clothes_dataset/best.onnx nếu chưa copy vào web/)
        rel = req_path.lstrip("/")
        target = (WEB_DIR / rel).resolve()
        if self._is_within(target, WEB_DIR) and target.is_file():
            self._serve_file(target)
            return

        if rel.endswith(".onnx"):
            candidates = [
                ROOT_DIR / Path(rel).name,
                DATASET_DIR / Path(rel).name,
                DATASET_DIR / "best.onnx",
            ]
            for cand in candidates:
                resolved = cand.resolve()
                if self._is_within(resolved, ROOT_DIR) and resolved.is_file():
                    self._serve_file(resolved)
                    return

        self.send_error(404, "Not found")

    def _serve_dataset_manifest(self) -> None:
        manifest_file = DATASET_DIR / "manifest.json"
        if not manifest_file.is_file():
            payload = json.dumps({"items": []}).encode("utf-8")
        else:
            payload = manifest_file.read_bytes()

        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def _serve_file(self, file_path: Path) -> None:
        ctype, _ = mimetypes.guess_type(str(file_path))
        if not ctype:
            ctype = "application/octet-stream"

        data = file_path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if file_path.suffix.lower() in {".html", ".js", ".webmanifest", ".json"}:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    @staticmethod
    def _is_within(target: Path, base: Path) -> bool:
        try:
            target.relative_to(base.resolve())
            return True
        except ValueError:
            return False


def create_https_server(host: str, port: int, cert_path: Path, key_path: Path) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, port), TryOnRequestHandler)
    ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ssl_ctx.load_cert_chain(certfile=str(cert_path), keyfile=str(key_path))
    httpd.socket = ssl_ctx.wrap_socket(httpd.socket, server_side=True)
    return httpd


def main() -> None:
    parser = argparse.ArgumentParser(description="HTTPS server cho iPhone Safari Virtual Try-On")
    parser.add_argument("--host", default="0.0.0.0", help="Host lắng nghe (mặc định 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8443, help="Cổng HTTPS (mặc định 8443)")
    parser.add_argument("--cert-only", action="store_true", help="Chỉ tạo chứng chỉ SSL rồi thoát")
    args = parser.parse_args()

    lan_ip = detect_lan_ip()
    cert_path, key_path = ensure_self_signed_cert(lan_ip)

    if args.cert_only:
        print(f"SSL ready: cert={cert_path}, key={key_path}, lan_ip={lan_ip}")
        return

    httpd = create_https_server(args.host, args.port, cert_path, key_path)
    print("\n=== Rust AI Virtual Try-On (iPhone Safari HTTPS Server) ===")
    print(f"Local URL : https://127.0.0.1:{args.port}")
    print(f"iPhone URL: https://{lan_ip}:{args.port}")
    print("\nHướng dẫn mở trên iPhone (không cần máy Mac):")
    print("  1. Kết nối iPhone và máy Windows vào CÙNG một mạng Wi-Fi.")
    print(f"  2. Mở Safari trên iPhone và nhập: https://{lan_ip}:{args.port}")
    print("  3. Khi Safari báo kết nối chưa tin cậy -> bấm 'Hiển thị chi tiết' -> 'Truy cập trang web này'.")
    print("  4. Bấm 'Bật Camera & Bắt đầu' để thử quần áo real-time (hoặc bấm Chia sẻ -> Thêm vào MH chính).\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nĐã dừng server.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
