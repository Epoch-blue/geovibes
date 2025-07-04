"""
Utility functions for GeoVibes.
"""

import os
from typing import Dict, Any
from .ui_config.constants import DatabaseConstants


def diagnose_gcs_connection(gcs_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Diagnose GCS connection issues and provide debugging information.
    
    Args:
        gcs_path: GCS path to diagnose
        verbose: Whether to print diagnostic information
        
    Returns:
        Dictionary with diagnostic information
    """
    diagnosis = {
        'is_gcs_path': DatabaseConstants.is_gcs_path(gcs_path),
        'has_hmac_keys': False,
        'has_gcloud_auth': False,
        'connection_test': False,
        'error_message': None
    }
    
    if verbose:
        print(f"🔍 Diagnosing GCS connection for: {gcs_path}")
    
    # Check HMAC keys
    gcs_key_id = os.getenv('GCS_ACCESS_KEY_ID')
    gcs_secret = os.getenv('GCS_SECRET_ACCESS_KEY')
    diagnosis['has_hmac_keys'] = bool(gcs_key_id and gcs_secret)
    
    if verbose:
        if diagnosis['has_hmac_keys']:
            print("✅ HMAC keys found in environment")
        else:
            print("❌ HMAC keys not found in environment")
    
    # Check gcloud authentication
    try:
        import subprocess
        result = subprocess.run(['gcloud', 'auth', 'list'], 
                              capture_output=True, text=True, timeout=10)
        diagnosis['has_gcloud_auth'] = result.returncode == 0 and 'ACTIVE' in result.stdout
        if verbose:
            if diagnosis['has_gcloud_auth']:
                print("✅ gcloud authentication available")
            else:
                print("❌ gcloud authentication not available")
    except Exception as e:
        diagnosis['has_gcloud_auth'] = False
        if verbose:
            print(f"❌ gcloud check failed: {e}")
    
    # Test connection
    if diagnosis['is_gcs_path']:
        try:
            conn = DatabaseConstants.setup_duckdb_connection(gcs_path, read_only=True)
            # Try a simple query
            result = conn.execute("SELECT 1").fetchone()
            diagnosis['connection_test'] = result is not None
            conn.close()
            if verbose:
                print("✅ Connection test successful")
        except Exception as e:
            diagnosis['connection_test'] = False
            diagnosis['error_message'] = str(e)
            if verbose:
                print(f"❌ Connection test failed: {e}")
    
    return diagnosis


def print_gcs_setup_help():
    """Print helpful GCS setup instructions."""
    print("\n🔧 GCS Setup Help:")
    print("1. Create HMAC keys at: https://console.cloud.google.com/storage/settings")
    print("2. Set environment variables:")
    print("   export GCS_ACCESS_KEY_ID='your_key_here'")
    print("   export GCS_SECRET_ACCESS_KEY='your_secret_here'")
    print("3. Or use gcloud authentication:")
    print("   gcloud auth application-default login")
    print("4. See GCS_SETUP.md for detailed instructions")
    print() 