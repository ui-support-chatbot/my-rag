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
        # Default test for Ollama on localhost, host bridge, and public IP
        localhost_ip = "127.0.0.1"
        host_ip = "172.17.0.1"
        server_ip = "152.118.31.54"
        
        print("--- Testing Localhost (Host Network) ---")
        test_connection(f"http://{localhost_ip}:11434/api/tags")

        print("\n--- Testing Bridge IP ---")
        test_connection(f"http://{host_ip}:11434/api/tags")
        
        print("\n--- Testing Server IP ---")
        test_connection(f"http://{server_ip}:11434/api/tags")
    else:
        test_connection(sys.argv[1])
