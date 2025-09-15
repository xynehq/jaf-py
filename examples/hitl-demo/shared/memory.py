"""
Memory provider setup for HITL demos.

This module provides utilities for setting up memory providers from environment configuration.
"""

import os
from typing import Optional, Any

from jaf.memory.factory import create_memory_provider_from_env
from jaf.memory.types import MemoryProvider


# Color utilities for console output
class Colors:
    GREEN = '\x1b[32m'
    YELLOW = '\x1b[33m'
    CYAN = '\x1b[36m'
    DIM = '\x1b[2m'
    RESET = '\x1b[0m'
    
    @classmethod
    def green(cls, text: str) -> str:
        return f"{cls.GREEN}{text}{cls.RESET}"
    
    @classmethod
    def yellow(cls, text: str) -> str:
        return f"{cls.YELLOW}{text}{cls.RESET}"
    
    @classmethod
    def cyan(cls, text: str) -> str:
        return f"{cls.CYAN}{text}{cls.RESET}"
    
    @classmethod
    def dim(cls, text: str) -> str:
        return f"{cls.DIM}{text}{cls.RESET}"


async def setup_memory_provider() -> MemoryProvider:
    """Setup memory provider from environment configuration."""
    print(Colors.cyan('💾 Setting up memory provider...'))
    
    memory_type = os.getenv('JAF_MEMORY_TYPE', 'memory').lower()
    
    if memory_type == 'redis':
        try:
            # Check Redis connection is available
            import redis.asyncio as redis
            redis.Redis(
                host=os.getenv('JAF_REDIS_HOST', 'localhost'),
                port=int(os.getenv('JAF_REDIS_PORT', '6379')),
                db=int(os.getenv('JAF_REDIS_DB', '0'))
            )
            
            result = await create_memory_provider_from_env()
            if hasattr(result, 'data'):
                memory_provider = result.data
            else:
                memory_provider = result
            print(Colors.green('✅ Redis memory provider initialized'))
            return memory_provider
                
        except Exception as e:
            print(Colors.yellow(f'⚠️  Redis not available, falling back to in-memory'))
            print(Colors.dim(f'   Error: {str(e)}'))
            result = await create_memory_provider_from_env()
            if hasattr(result, 'data'):
                memory_provider = result.data
            else:
                memory_provider = result
            print(Colors.green('✅ In-memory provider initialized (fallback)'))
            return memory_provider
                
    elif memory_type == 'postgres':
        try:
            # Create PostgreSQL client if PostgreSQL is configured
            import asyncpg
            await asyncpg.connect(
                host=os.getenv('JAF_POSTGRES_HOST', 'localhost'),
                port=int(os.getenv('JAF_POSTGRES_PORT', '5432')),
                database=os.getenv('JAF_POSTGRES_DB', 'jaf_memory'),
                user=os.getenv('JAF_POSTGRES_USER', 'postgres'),
                password=os.getenv('JAF_POSTGRES_PASSWORD'),
                ssl='require' if os.getenv('JAF_POSTGRES_SSL') == 'true' else 'prefer'
            )
            
            print(Colors.dim('   Connected to PostgreSQL'))
            
            result = await create_memory_provider_from_env()
            if hasattr(result, 'data'):
                memory_provider = result.data
            else:
                memory_provider = result
            print(Colors.green('✅ PostgreSQL memory provider initialized'))
            return memory_provider
                
        except Exception as e:
            print(Colors.yellow(f'⚠️  PostgreSQL not available, falling back to in-memory'))
            print(Colors.dim(f'   Error: {str(e)}'))
            if hasattr(e, 'cause') and e.cause:
                print(Colors.dim(f'   Cause: {str(e.cause)}'))
            result = await create_memory_provider_from_env()
            if hasattr(result, 'data'):
                memory_provider = result.data
            else:
                memory_provider = result
            print(Colors.green('✅ In-memory provider initialized (fallback)'))
            return memory_provider
    else:
        # In-memory provider
        result = await create_memory_provider_from_env()
        if hasattr(result, 'data'):
            memory_provider = result.data
        else:
            memory_provider = result
        print(Colors.green('✅ In-memory provider initialized'))
        return memory_provider