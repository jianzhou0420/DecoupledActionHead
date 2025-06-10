#!/usr/bin/env python3
import os
import socks
import socket
import ssl
import wandb
import socks
import socket
import ssl
from urllib.request import Request, urlopen

IP_ADDR = '127.0.0.1'
PORT=1080
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
socks.set_default_proxy(socks.SOCKS5, IP_ADDR, PORT)
socket.socket = socks.socksocket


project_name = "ssh-proxy-test"
entity_name = None  # replace with your wandb entity if needed

run = wandb.init(project=project_name, entity=entity_name, reinit=True)
run.finish()

# If we reach here, WandB was able to talk to api.wandb.ai through the proxy.
print("🎉 WandB connectivity test passed!")

