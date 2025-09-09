#!/usr/bin/env python3
"""
Setup script for PhotoStudio Pipeline MCP integration

This script helps set up Model Context Protocol (MCP) servers for the 
JSON Ghost-Mannequin Pipeline to work with Warp agents.
"""

import json
import os
import subprocess
import sys
from pathlib import Path


def install_mcp_dependencies():
    """Install required MCP server dependencies."""
    print("📦 Installing MCP server dependencies...")
    
    try:
        # Install MCP server dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "-r", "mcp_server/requirements.txt"
        ])
        print("✅ MCP dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install MCP dependencies: {e}")
        return False
    
    return True


def validate_pipeline_dependencies():
    """Check if pipeline dependencies are available."""
    print("🔍 Validating pipeline dependencies...")
    
    required_modules = [
        "src.photostudio.steps.step1_preprocess",
        "src.photostudio.steps.step4_qa_corrector"
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print(f"⚠️  Missing pipeline modules: {', '.join(missing_modules)}")
        print("💡 Install the main pipeline with: pip install -e .")
        return False
    
    print("✅ Pipeline dependencies validated")
    return True


def setup_environment_variables():
    """Guide user through setting up environment variables."""
    print("\n🔧 Environment Variables Setup")
    print("=" * 50)
    
    env_vars = {
        "GITHUB_TOKEN": "GitHub Personal Access Token for repository access",
        "GOOGLE_API_KEY": "Google API key for Gemini integration", 
        "HF_TOKEN": "Hugging Face token for SDXL model access"
    }
    
    env_file_path = Path(".env.mcp")
    env_content = []
    
    for var, description in env_vars.items():
        current_value = os.environ.get(var, "")
        print(f"\n{var}: {description}")
        
        if current_value:
            print(f"  Current value: {'*' * len(current_value[:4])}...")
            use_current = input("  Use current environment value? (y/n): ").lower().strip()
            if use_current == 'y':
                env_content.append(f"{var}={current_value}")
                continue
        
        new_value = input(f"  Enter {var} (or skip): ").strip()
        if new_value:
            env_content.append(f"{var}={new_value}")
    
    if env_content:
        with open(env_file_path, "w") as f:
            f.write("\n".join(env_content))
        print(f"\n✅ Environment variables saved to {env_file_path}")
        print("💡 Add 'source .env.mcp' to your shell profile to load automatically")
    
    return True


def generate_warp_config():
    """Generate Warp MCP configuration instructions."""
    print("\n🚀 Warp Configuration Setup")
    print("=" * 50)
    
    config_path = Path("mcp_config.json")
    
    with open(config_path) as f:
        config = json.load(f)
    
    print("\n1. Open Warp and navigate to:")
    print("   Settings > AI > Manage MCP servers")
    print("   OR")
    print("   Command Palette (Ctrl+K) > 'Open MCP Servers'")
    
    print("\n2. Click '+ Add' and paste this configuration:")
    print("\n" + "="*60)
    print(json.dumps(config, indent=2))
    print("="*60)
    
    print("\n3. Save and start the 'photostudio-pipeline' server")
    print("\n4. You should see these tools available to Warp agents:")
    tools = [
        "analyze-garment-quality - Quality assessment for garment images",
        "get-pipeline-status - Real-time pipeline monitoring", 
        "recommend-parameters - Smart parameter suggestions",
        "debug-pipeline-step - Step-by-step debugging assistance",
        "optimize-route-selection - SDXL vs Gemini route optimization"
    ]
    
    for tool in tools:
        print(f"   • {tool}")
    
    return True


def test_mcp_server():
    """Test if the MCP server can start properly."""
    print("\n🧪 Testing MCP server startup...")
    
    try:
        # Test import of the MCP server module
        sys.path.insert(0, str(Path.cwd()))
        from mcp_server.photostudio_mcp import server
        print("✅ MCP server module imports successfully")
        
        # Test server initialization
        if server:
            print("✅ MCP server initializes correctly")
            return True
            
    except Exception as e:
        print(f"❌ MCP server test failed: {e}")
        print("💡 Check that all dependencies are installed correctly")
        return False


def main():
    """Main setup function."""
    print("🎨 PhotoStudio Pipeline MCP Setup")
    print("=" * 50)
    print("This script will help you set up MCP integration with Warp agents")
    print()
    
    # Create MCP server directory if it doesn't exist
    mcp_dir = Path("mcp_server")
    mcp_dir.mkdir(exist_ok=True)
    
    steps = [
        ("Installing MCP dependencies", install_mcp_dependencies),
        ("Validating pipeline dependencies", validate_pipeline_dependencies), 
        ("Setting up environment variables", setup_environment_variables),
        ("Generating Warp configuration", generate_warp_config),
        ("Testing MCP server", test_mcp_server)
    ]
    
    failed_steps = []
    
    for step_name, step_func in steps:
        print(f"\n📋 {step_name}...")
        try:
            success = step_func()
            if not success:
                failed_steps.append(step_name)
        except Exception as e:
            print(f"❌ {step_name} failed: {e}")
            failed_steps.append(step_name)
    
    print("\n" + "="*50)
    if failed_steps:
        print(f"⚠️  Setup completed with {len(failed_steps)} issues:")
        for step in failed_steps:
            print(f"   • {step}")
        print("\n💡 Check the error messages above and retry failed steps")
    else:
        print("🎉 MCP setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Restart Warp terminal")
        print("2. Add the MCP configuration shown above")
        print("3. Start the photostudio-pipeline MCP server")
        print("4. Ask Warp agents about your pipeline!")
        print("\n💬 Example questions to try:")
        print('   • "Analyze the quality of this garment image"')
        print('   • "What\'s the current status of the pipeline?"')
        print('   • "Recommend parameters for processing a silk dress"')
        print('   • "Debug issues with Step 3 rendering"')


if __name__ == "__main__":
    main()
