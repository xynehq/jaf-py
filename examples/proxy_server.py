"""
Simple HTTP proxy server for testing JAF proxy functionality.

This creates a basic HTTP proxy server that logs requests and forwards them.
Use this for testing proxy support without needing external proxy infrastructure.
"""

import asyncio
import logging
import socket
import sys
from typing import Optional
import aiohttp
from aiohttp import web, ClientSession
import argparse


class SimpleProxyServer:
    """Simple HTTP proxy server for testing."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 8888, auth_required: bool = False):
        self.host = host
        self.port = port
        self.auth_required = auth_required
        self.valid_credentials = {"testuser": "testpass", "admin": "secret"}
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ProxyServer")
    
    def check_auth(self, request: web.Request) -> bool:
        """Check proxy authentication if required."""
        if not self.auth_required:
            return True
        
        auth_header = request.headers.get('Proxy-Authorization')
        if not auth_header:
            return False
        
        try:
            # Parse "Basic base64encoded" format
            auth_type, credentials = auth_header.split(' ', 1)
            if auth_type.lower() != 'basic':
                return False
            
            import base64
            decoded = base64.b64decode(credentials).decode('utf-8')
            username, password = decoded.split(':', 1)
            
            return self.valid_credentials.get(username) == password
        except Exception:
            return False
    
    async def handle_request(self, request: web.Request) -> web.Response:
        """Handle HTTP proxy requests."""
        # Check authentication
        if not self.check_auth(request):
            self.logger.warning(f"Unauthorized proxy request from {request.remote}")
            return web.Response(
                status=407,
                headers={'Proxy-Authenticate': 'Basic realm="Proxy"'},
                text="Proxy Authentication Required"
            )
        
        # Extract target URL
        url = str(request.url)
        method = request.method
        headers = dict(request.headers)
        
        # Remove proxy-specific headers
        headers.pop('Proxy-Authorization', None)
        headers.pop('Proxy-Connection', None)
        
        self.logger.info(f"Proxying {method} request to {url}")
        
        try:
            # Forward the request
            async with ClientSession() as session:
                async with session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    data=await request.read()
                ) as response:
                    # Forward response
                    response_headers = dict(response.headers)
                    response_headers.pop('Transfer-Encoding', None)  # Let aiohttp handle this
                    
                    body = await response.read()
                    
                    self.logger.info(f"Response: {response.status} for {url}")
                    
                    return web.Response(
                        status=response.status,
                        headers=response_headers,
                        body=body
                    )
        
        except Exception as e:
            self.logger.error(f"Error proxying request to {url}: {e}")
            return web.Response(
                status=502,
                text=f"Proxy Error: {str(e)}"
            )
    
    async def handle_connect(self, request: web.Request) -> web.Response:
        """Handle HTTPS CONNECT requests."""
        # Check authentication
        if not self.check_auth(request):
            return web.Response(
                status=407,
                headers={'Proxy-Authenticate': 'Basic realm="Proxy"'},
                text="Proxy Authentication Required"
            )
        
        target = request.path_qs
        self.logger.info(f"CONNECT request to {target}")
        
        try:
            host, port = target.split(':')
            port = int(port)
        except ValueError:
            return web.Response(status=400, text="Bad CONNECT request")
        
        try:
            # Create connection to target
            reader, writer = await asyncio.open_connection(host, port)
            
            # Send 200 Connection established
            response = web.StreamResponse(status=200, reason='Connection established')
            await response.prepare(request)
            
            # Start tunneling
            async def tunnel_data(source, dest):
                try:
                    while True:
                        data = await source.read(4096)
                        if not data:
                            break
                        dest.write(data)
                        await dest.drain()
                except Exception:
                    pass
                finally:
                    dest.close()
            
            # Start bidirectional tunneling
            await asyncio.gather(
                tunnel_data(request.transport, writer),
                tunnel_data(reader, request.transport),
                return_exceptions=True
            )
            
        except Exception as e:
            self.logger.error(f"Error in CONNECT to {target}: {e}")
            return web.Response(status=502, text=f"Connect Error: {str(e)}")
        
        return response
    
    async def start_server(self):
        """Start the proxy server."""
        app = web.Application()
        
        # Handle CONNECT method for HTTPS
        app.router.add_route('CONNECT', '/{path:.*}', self.handle_connect)
        
        # Handle all other HTTP methods
        for method in ['GET', 'POST', 'PUT', 'DELETE', 'HEAD', 'OPTIONS', 'PATCH']:
            app.router.add_route(method, '/{path:.*}', self.handle_request)
        
        runner = web.AppRunner(app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        auth_info = " (with auth)" if self.auth_required else " (no auth)"
        self.logger.info(f"Proxy server started on {self.host}:{self.port}{auth_info}")
        
        if self.auth_required:
            self.logger.info("Valid credentials:")
            for username, password in self.valid_credentials.items():
                self.logger.info(f"  {username}:{password}")
        
        return runner


async def run_proxy_server(host: str, port: int, auth: bool = False):
    """Run the proxy server."""
    proxy = SimpleProxyServer(host, port, auth)
    runner = await proxy.start_server()
    
    print(f"\n🚀 Proxy server running on http://{host}:{port}")
    if auth:
        print("📋 Authentication required:")
        print("   Username: testuser, Password: testpass")
        print("   Username: admin, Password: secret")
    else:
        print("🔓 No authentication required")
    
    print("\n📝 Test configuration:")
    print(f"   HTTP_PROXY=http://{host}:{port}")
    print(f"   HTTPS_PROXY=http://{host}:{port}")
    if auth:
        print("   PROXY_USERNAME=testuser")
        print("   PROXY_PASSWORD=testpass")
    
    print("\n🛑 Press Ctrl+C to stop the server")
    
    try:
        # Keep the server running
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Stopping proxy server...")
        await runner.cleanup()


def main():
    """Main function to start proxy servers."""
    parser = argparse.ArgumentParser(description="Simple HTTP Proxy Server for JAF Testing")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8888, help="Port to bind to (default: 8888)")
    parser.add_argument("--auth", action="store_true", help="Require proxy authentication")
    parser.add_argument("--multiple", action="store_true", help="Start multiple proxy servers")
    
    args = parser.parse_args()
    
    if args.multiple:
        print("🚀 Starting multiple proxy servers for testing...")
        print("This will start:")
        print("  1. Proxy without auth on port 8888")
        print("  2. Proxy with auth on port 8889")
        print("  3. Additional proxy on port 8890")
        
        async def run_multiple():
            tasks = [
                run_proxy_server("127.0.0.1", 8888, False),
                run_proxy_server("127.0.0.1", 8889, True),
                run_proxy_server("127.0.0.1", 8890, False),
            ]
            await asyncio.gather(*tasks, return_exceptions=True)
        
        asyncio.run(run_multiple())
    else:
        asyncio.run(run_proxy_server(args.host, args.port, args.auth))


if __name__ == "__main__":
    # Check if aiohttp is available
    try:
        import aiohttp
    except ImportError:
        print("❌ aiohttp is required to run the proxy server")
        print("Install it with: pip install aiohttp")
        sys.exit(1)
    
    main()