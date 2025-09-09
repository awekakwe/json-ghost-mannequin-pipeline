# 🚀 Warp MCP Setup Guide

Your MCP servers are ready! Follow these steps to integrate them with Warp agents.

## Step 1: Open Warp MCP Settings

**Method 1:** Settings Menu
- Open Warp → Settings → AI → Manage MCP servers

**Method 2:** Command Palette  
- Press `Ctrl+K` (or `Cmd+K` on Mac)
- Type "Open MCP Servers" and select it

## Step 2: Add MCP Configuration

1. Click the **"+ Add"** button in the MCP servers page
2. **Copy and paste this EXACT configuration:**

```json
{
  "mcpServers": {
    "photostudio-pipeline": {
      "command": "python",
      "args": ["mcp_server/photostudio_mcp.py"],
      "working_directory": "/Users/Peter/Projects/json-ghost-mannequin-pipeline",
      "env": {
        "PYTHONPATH": "/Users/Peter/Projects/json-ghost-mannequin-pipeline",
        "LOG_LEVEL": "INFO"
      },
      "start_on_launch": true
    },
    "github": {
      "command": "npx",
      "args": ["-y", "mcp-remote", "https://api.githubcopilot.com/mcp/"],
      "start_on_launch": true
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/Users/Peter/Projects/json-ghost-mannequin-pipeline"],
      "start_on_launch": true
    }
  }
}
```

3. Click **"Save"** or **"Add Servers"**

## Step 3: Start the Servers

After adding the configuration, you should see three servers:

1. **photostudio-pipeline** 🟢 (should start automatically)
2. **github** 🟢 (should start automatically)  
3. **filesystem** 🟢 (should start automatically)

If any show as 🔴 (stopped), click the **"Start"** button next to them.

## Step 4: Verify Everything Works

Once all servers show 🟢 (running), try these test commands with Warp agents:

### Test Pipeline Integration
```bash
# Ask: "What's the current status of my photostudio pipeline?"
# Expected: Dashboard with step health, queue info, and metrics
```

### Test GitHub Integration  
```bash
# Ask: "Show me recent commits in this repository"
# Expected: List of recent commits with details
```

### Test File System Access
```bash
# Ask: "List the files in my pipeline project"
# Expected: Directory listing with project files
```

## 🎯 Available Tools

Once setup is complete, Warp agents will have access to these powerful tools:

### **Pipeline-Specific Tools:**
- `analyze-garment-quality` - Analyze image quality metrics
- `get-pipeline-status` - Real-time step monitoring  
- `recommend-parameters` - Smart parameter suggestions
- `debug-pipeline-step` - Troubleshooting assistance
- `optimize-route-selection` - SDXL vs Gemini optimization

### **GitHub Tools:**
- Repository browsing and file access
- Commit history and branch information
- Issue and PR management
- Repository statistics and insights

### **Filesystem Tools:**
- Direct file and directory access
- File content reading and analysis
- Project structure exploration
- Configuration file management

## 🚨 Troubleshooting

### Server Won't Start
- Check that you're in the correct directory: `/Users/Peter/Projects/json-ghost-mannequin-pipeline`
- Verify Python environment is activated
- Click "View Logs" in Warp to see error details

### Permission Errors
- Ensure your GitHub token has the required permissions
- Check file permissions in the project directory
- Verify environment variables are loaded: `source .env.mcp`

### Tools Not Appearing
- Restart Warp completely after adding servers
- Check that all servers show 🟢 (running) status
- Try stopping and starting individual servers

## 🎉 Success Indicators

You'll know everything is working when:

✅ All 3 MCP servers show 🟢 running status  
✅ Warp agents respond to pipeline status questions  
✅ GitHub repository information is accessible  
✅ File system commands work properly  

## 💬 Example Agent Conversations

Once setup, try these natural language commands:

**Pipeline Monitoring:**
- "What's the health status of all pipeline steps?"
- "Show me the processing queue and active jobs"
- "Are there any failed processes I should know about?"

**Quality Analysis:**
- "Analyze the quality of test_garment.jpg"
- "What are the current quality gate thresholds?"
- "Show me recent quality trends"

**Parameter Optimization:**
- "Recommend processing parameters for a silk dress"
- "Which route should I use for cotton shirts?"
- "What auto-tuning adjustments were made recently?"

**GitHub Integration:**
- "Show me the latest commits to this repository"
- "What issues are currently open?"
- "Create a summary of recent development activity"

**File Management:**
- "List all configuration files in the project"
- "Show me the contents of config/style_kit.yml"
- "What test images are available?"

Your sophisticated AI photography pipeline is now fully integrated with Warp agents! 🎨✨
