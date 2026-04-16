import sys
import urllib.request
import json
import socket

def test_connection(url):
    print(f"[*] Testing connection to: {url}")
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            status = response.getcode()
            content = response.read().decode('utf-8')
            print(f"[+] SUCCESS! Status: {status}")
            print(f"[+] Response: {content[:200]}...")
            return True
    except Exception as e:
        print(f"[-] FAILED! Error: {e}")
        return False

def check_hostname(hostname):
    print(f"[*] Checking hostname resolution for: {hostname}")
    try:
        ip = socket.gethostbyname(hostname)
        print(f"[+] {hostname} resolved to: {ip}")
    except Exception as e:
        print(f"[-] Failed to resolve {hostname}: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        # Default test for Ollama on host bridge
        host_ip = "172.17.0.1"
        check_hostname("host.docker.internal")
        test_connection(f"http://{host_ip}:11434/api/tags")
    else:
        test_connection(sys.argv[1])
