"""Local HTTPS for Safari camera access. Only the server leaf private key is kept."""

import ipaddress
import ssl
from datetime import datetime, timedelta, timezone

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.x509.oid import ExtendedKeyUsageOID, NameOID


def camera_tls(folder, ip):
    address = ipaddress.IPv4Address(ip)
    folder.mkdir(parents=True, exist_ok=True)
    ca_path = folder / "camera-root.cer"
    cert_path = folder / "camera.pem"
    key_path = folder / "camera.key"
    now = datetime.now(timezone.utc)
    leaf = x509.load_pem_x509_certificate(cert_path.read_bytes()) if cert_path.is_file() else None
    if (not ca_path.is_file() or not key_path.is_file() or leaf is None
            or x509.ExtensionOID.AUTHORITY_KEY_IDENTIFIER not in {e.oid for e in leaf.extensions}
            or leaf.not_valid_after_utc < now + timedelta(days=1)):
        def builder(name, issuer, key):
            return (x509.CertificateBuilder().subject_name(name).issuer_name(issuer)
                    .public_key(key.public_key()).serial_number(x509.random_serial_number())
                    .not_valid_before(now - timedelta(minutes=10)).not_valid_after(now + timedelta(days=90)))

        def name(value):
            return x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, value)])

        root_key = ec.generate_private_key(ec.SECP256R1())
        root_name = name("Twin Depot Camera " + ip)
        root = (builder(root_name, root_name, root_key)
                .add_extension(x509.SubjectKeyIdentifier.from_public_key(root_key.public_key()), False)
                .add_extension(x509.BasicConstraints(ca=True, path_length=0), True)
                .add_extension(x509.KeyUsage(False, False, False, False, False, True, True, False, False), True)
                .sign(root_key, hashes.SHA256()))
        key = ec.generate_private_key(ec.SECP256R1())
        leaf = (builder(name(ip), root_name, key)
                .add_extension(x509.AuthorityKeyIdentifier.from_issuer_public_key(root_key.public_key()), False)
                .add_extension(x509.SubjectKeyIdentifier.from_public_key(key.public_key()), False)
                .add_extension(x509.BasicConstraints(ca=False, path_length=None), True)
                .add_extension(x509.SubjectAlternativeName([x509.IPAddress(address)]), False)
                .add_extension(x509.ExtendedKeyUsage([ExtendedKeyUsageOID.SERVER_AUTH]), False)
                .add_extension(x509.KeyUsage(True, False, False, False, False, False, False, False, False), True)
                .sign(root_key, hashes.SHA256()))
        ca_path.write_bytes(root.public_bytes(serialization.Encoding.DER))
        cert_path.write_bytes(leaf.public_bytes(serialization.Encoding.PEM))
        key_path.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                               serialization.NoEncryption()))
        key_path.chmod(0o600)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.minimum_version = ssl.TLSVersion.TLSv1_2
    context.load_cert_chain(cert_path, key_path)
    return context, ca_path
