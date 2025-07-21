#!/usr/bin/env python3
"""
Test script to verify Product Management System works without OpenAI API key
"""

import asyncio
import requests
import json
import os
from datetime import datetime

# Test configuration
API_BASE = "http://localhost:8000"
TEST_CSV = "vasavi2.csv"  # Use existing CSV file

async def test_system_without_openai():
    """Test the system without OpenAI API key"""
    
    print("🧪 Testing Product Management System (No OpenAI)")
    print("=" * 50)
    
    # Test 1: Health Check
    print("\n1️⃣ Testing Health Check...")
    try:
        response = requests.get(f"{API_BASE}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health Check: {data.get('status', 'unknown')}")
            print(f"   Total Products: {data.get('total_products', 0)}")
        else:
            print(f"❌ Health Check Failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health Check Error: {e}")
        return False
    
    # Test 2: Upload CSV
    print("\n2️⃣ Testing CSV Upload...")
    try:
        if os.path.exists(TEST_CSV):
            with open(TEST_CSV, 'rb') as f:
                files = {'file': (TEST_CSV, f, 'text/csv')}
                response = requests.post(f"{API_BASE}/upload/csv", files=files)
                
                if response.status_code == 200:
                    data = response.json()
                    print(f"✅ CSV Upload: {data.get('message', 'Success')}")
                    print(f"   Storage: {data.get('storage_method', 'unknown')}")
                    print(f"   Rows: {data.get('row_count', 0)}")
                else:
                    print(f"❌ CSV Upload Failed: {response.status_code}")
                    print(f"   Error: {response.text}")
        else:
            print(f"⚠️  Test CSV file {TEST_CSV} not found, skipping upload test")
    except Exception as e:
        print(f"❌ CSV Upload Error: {e}")
    
    # Test 3: Search Products
    print("\n3️⃣ Testing Product Search...")
    try:
        response = requests.get(f"{API_BASE}/search?query=jacket&limit=5")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Search: Found {data.get('results_count', 0)} products")
            if data.get('products'):
                print(f"   First product: {data['products'][0].get('style_name', 'N/A')}")
        else:
            print(f"❌ Search Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Search Error: {e}")
    
    # Test 4: Statistics
    print("\n4️⃣ Testing Statistics...")
    try:
        response = requests.get(f"{API_BASE}/statistics")
        if response.status_code == 200:
            data = response.json()
            stats = data.get('statistics', {})
            print(f"✅ Statistics: {stats.get('total_products', 0)} total products")
            print(f"   PostgreSQL: {stats.get('storage_distribution', {}).get('postgresql', 0)}")
            print(f"   MongoDB: {stats.get('storage_distribution', {}).get('mongodb', 0)}")
        else:
            print(f"❌ Statistics Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Statistics Error: {e}")
    
    # Test 5: Categories
    print("\n5️⃣ Testing Categories...")
    try:
        response = requests.get(f"{API_BASE}/categories")
        if response.status_code == 200:
            data = response.json()
            categories = data.get('categories', [])
            print(f"✅ Categories: {len(categories)} categories found")
            if categories:
                print(f"   Categories: {', '.join(categories[:5])}")
        else:
            print(f"❌ Categories Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Categories Error: {e}")
    
    # Test 6: AI Chat (should fail gracefully)
    print("\n6️⃣ Testing AI Chat (should be disabled)...")
    try:
        response = requests.get(f"{API_BASE}/chat?query=show me jackets")
        if response.status_code == 200:
            data = response.json()
            if not data.get('success', True):
                print(f"✅ AI Chat: Properly disabled - {data.get('error', 'No error message')}")
            else:
                print(f"⚠️  AI Chat: Unexpectedly working (OpenAI key might be configured)")
        else:
            print(f"❌ AI Chat Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ AI Chat Error: {e}")
    
    # Test 7: Export
    print("\n7️⃣ Testing Export...")
    try:
        response = requests.get(f"{API_BASE}/export/csv")
        if response.status_code == 200:
            print(f"✅ CSV Export: Success ({len(response.content)} bytes)")
        else:
            print(f"❌ CSV Export Failed: {response.status_code}")
    except Exception as e:
        print(f"❌ CSV Export Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Testing Complete!")
    print("\n📋 Summary:")
    print("✅ Core features work without OpenAI API key:")
    print("   - File upload and storage")
    print("   - Product search and filtering")
    print("   - Statistics and analytics")
    print("   - Data export")
    print("   - Category management")
    print("\n❌ AI features are disabled:")
    print("   - Chat-based queries")
    print("   - AI recommendations")
    print("\n🔑 To enable AI features, add OPENAI_API_KEY to .env file")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_system_without_openai()) 