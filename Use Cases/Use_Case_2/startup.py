"""
Quick Start Script for Granular Document Editor
Run this script to verify your setup and start editing
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_environment():
    """Check if all required environment variables are set"""
    print("🔍 Checking environment setup...")
    print("="*60)
    
    # Check for API key (multiple possible names)
    api_key = (os.getenv("AZURE_OPENAI_API_KEY") or 
               os.getenv("AZURE_OPENAI_KEY"))
    
    # Check for endpoint
    endpoint = (os.getenv("AZURE_OPENAI_ENDPOINT") or 
                os.getenv("AZURE_OPENAI_BASE"))
    
    # Check for deployment name
    deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT") or
                  os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or
                  os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or
                  "gpt-4o-mini")  # Your default
    
    missing = []
    
    if not api_key or api_key.startswith("your-"):
        missing.append("AZURE_OPENAI_API_KEY")
        print(f"❌ AZURE_OPENAI_API_KEY: Not set or using placeholder")
    else:
        masked = api_key[:8] + "..." + api_key[-4:] if len(api_key) > 12 else "***"
        print(f"✓ AZURE_OPENAI_API_KEY: {masked}")
    
    if not endpoint or endpoint.startswith("https://your-"):
        missing.append("AZURE_OPENAI_ENDPOINT")
        print(f"❌ AZURE_OPENAI_ENDPOINT: Not set or using placeholder")
    else:
        print(f"✓ AZURE_OPENAI_ENDPOINT: {endpoint}")
    
    print(f"✓ AZURE_OPENAI_DEPLOYMENT: {deployment}")
    
    print("="*60)
    
    if missing:
        print("\n⚠️  Missing configuration:")
        print("\nPlease update your .env file with:")
        for var in missing:
            print(f"   {var}=your_actual_value")
        return False, None, None, None
    
    print("\n✅ Environment configured correctly!")
    return True, api_key, endpoint, deployment


def test_connection():
    """Test Azure OpenAI connection"""
    try:
        from openai import AzureOpenAI
        
        print("\n🔌 Testing Azure OpenAI connection...")
        print("="*60)
        
        success, api_key, endpoint, deployment = check_environment()
        if not success:
            return False
        
        # Remove quotes if present
        endpoint = endpoint.strip('"\'')
        api_key = api_key.strip('"\'')
        
        # Ensure endpoint ends with /
        if not endpoint.endswith('/'):
            endpoint = endpoint + '/'
        
        print(f"Using deployment: {deployment}")
        
        client = AzureOpenAI(
            api_key=api_key,
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
            azure_endpoint=endpoint
        )
        
        # Test with a simple completion
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": "Say 'Connection successful!'"}],
            max_tokens=10
        )
        
        print(f"✓ Connection successful!")
        print(f"✓ Model responded: {response.choices[0].message.content}")
        print("="*60)
        return True
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Connection failed: {error_msg}")
        print("\n🔍 Troubleshooting:")
        
        if "DeploymentNotFound" in error_msg:
            print("\n   Issue: Deployment not found")
            print("   Solutions:")
            print(f"   1. Check your deployment name: '{deployment}'")
            print("   2. Go to Azure OpenAI Studio → Deployments")
            print("   3. Verify the deployment name matches exactly")
            print("   4. Common deployment names:")
            print("      - gpt-4o-mini")
            print("      - gpt-4o")
            print("      - gpt-4")
            print("      - gpt-35-turbo")
            print("\n   Your .env should have:")
            print("   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini")
        elif "401" in error_msg or "Authentication" in error_msg:
            print("   1. Check your API key is correct")
            print("   2. Regenerate key in Azure Portal if needed")
        elif "404" in error_msg:
            print("   1. Verify your endpoint URL")
            print("   2. Check the resource name in the URL")
        else:
            print("   1. Check Azure OpenAI resource is active")
            print("   2. Verify region supports your model")
            print("   3. Check quota limits")
        
        print("="*60)
        return False


def show_current_config():
    """Show current configuration"""
    print("\n📋 Current Configuration:")
    print("="*60)
    
    api_key = os.getenv("AZURE_OPENAI_API_KEY", "").strip('"\'')
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip('"\'')
    deployment = (os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or
                  os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME") or
                  "gpt-4o-mini")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview")
    
    print(f"Endpoint:     {endpoint}")
    print(f"API Version:  {api_version}")
    print(f"Deployment:   {deployment}")
    if api_key:
        print(f"API Key:      {api_key[:8]}...{api_key[-4:]}")
    print("="*60)


def show_sample_commands():
    """Display sample commands to try"""
    print("\n📝 Sample Commands to Try:")
    print("="*60)
    print("""
1. Remove a section:
   "remove the competitive landscape section"

2. Add a new section:
   "add a section about Risk Assessment after Market Analysis"

3. Expand existing content:
   "expand the Recommendations section with more details"

4. Update section content:
   "update Executive Summary to mention our Q1 goals"

5. Batch operations:
   batch = BatchEditor(editor)
   batch.queue_edit("remove section X")
   batch.queue_edit("add section about Y")
   batch.execute_batch()
""")
    print("="*60)


def main():
    """Main startup routine"""
    print("\n" + "="*60)
    print("  GRANULAR DOCUMENT EDITOR - STARTUP CHECK")
    print("="*60 + "\n")
    
    # Show current config
    show_current_config()
    
    # Step 1: Check environment
    success, api_key, endpoint, deployment = check_environment()
    if not success:
        print("\n❌ Setup incomplete. Please configure your .env file.")
        return
    
    # Step 2: Test connection
    if not test_connection():
        print("\n❌ Connection test failed. Please check your configuration above.")
        print("\n💡 Quick Fix:")
        print("   Your .env file should include:")
        print("   AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o-mini")
        print("\n   Or update the notebook to use:")
        deployment_name = (os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME") or
                          os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"))
        print(f"   deployment_name=\"{deployment_name}\"")
        return
    
    # Step 3: Show next steps
    print("\n✅ All checks passed! Ready to use.")
    
    show_sample_commands()
    
    print("\n📚 Next Steps:")
    print("="*60)
    print("""
1. Open Jupyter Notebook:
   jupyter notebook granular_document_editor.ipynb

2. Update Cell 2 configuration:
   AZURE_CONFIG = {
       "api_key": os.getenv("AZURE_OPENAI_API_KEY"),
       "api_version": os.getenv("AZURE_OPENAI_API_VERSION", "2024-05-01-preview"),
       "azure_endpoint": os.getenv("AZURE_OPENAI_ENDPOINT"),
       "deployment_name": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini"),
       "deployment_name_mini": os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o-mini")
   }

3. Run all cells in order (Cell → Run All)

4. Try the example commands in cells 9-13

5. For interactive mode, run cell 19:
   interactive_mode_advanced(editor)
""")
    print("="*60)
    
    print("\n💡 Tips:")
    print("   • You're using gpt-4o-mini (fast and cheap!)")
    print("   • Batch similar operations to save tokens")
    print("   • Check coherence after major edits")
    print("   • Monitor costs with TokenAnalytics")
    
    print("\n🎯 Expected Performance:")
    print("   • 60-85% token reduction")
    print("   • 3-5x faster than full regeneration")
    print("   • Very low cost with gpt-4o-mini")
    
    print("\n" + "="*60)
    print("  Happy Editing! 🎉")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()