import hashlib
import os
from cryptography.fernet import Fernet
from typing import Any, Dict

class SecurityManager:
    def __init__(self):
        self.key = os.environ.get('ENCRYPTION_KEY') or Fernet.generate_key()
        self.cipher_suite = Fernet(self.key)

    def encrypt_data(self, data: str) -> bytes:
        return self.cipher_suite.encrypt(data.encode())

    def decrypt_data(self, encrypted_data: bytes) -> str:
        return self.cipher_suite.decrypt(encrypted_data).decode()

    def hash_data(self, data: str) -> str:
        return hashlib.sha256(data.encode()).hexdigest()

class PrivacyFilter:
    def __init__(self):
        self.sensitive_fields = ['name', 'email', 'phone', 'address', 'ssn']

    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        anonymized = {}
        for key, value in data.items():
            if key.lower() in self.sensitive_fields:
                anonymized[key] = self._mask_data(value)
            elif isinstance(value, dict):
                anonymized[key] = self.anonymize_data(value)
            else:
                anonymized[key] = value
        return anonymized

    def _mask_data(self, value: str) -> str:
        if len(value) <= 4:
            return '*' * len(value)
        return value[:2] + '*' * (len(value) - 4) + value[-2:]

# Usage
security_manager = SecurityManager()
privacy_filter = PrivacyFilter()

sensitive_data = {
    "name": "John Doe",
    "email": "john.doe@example.com",
    "message": "This is a secret message",
    "metadata": {
        "phone": "1234567890"
    }
}

# Anonymize sensitive data
anonymized_data = privacy_filter.anonymize_data(sensitive_data)

# Encrypt the anonymized data
encrypted_data = security_manager.encrypt_data(str(anonymized_data))

print("Anonymized data:", anonymized_data)
print("Encrypted data:", encrypted_data)

# Decryption (for authorized access)
decrypted_data = security_manager.decrypt_data(encrypted_data)
print("Decrypted data:", decrypted_data)