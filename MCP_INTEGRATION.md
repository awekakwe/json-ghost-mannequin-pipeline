# MCP Integration for JSON Ghost-Mannequin Pipeline

This document describes how to integrate the JSON Ghost-Mannequin Pipeline with Warp agents using Model Context Protocol (MCP) servers.

## What is MCP?

Model Context Protocol (MCP) is an open standard that allows AI agents to access external tools and data sources through a standardized interface. Think of MCP servers as plugins that extend what Warp agents can do.

For our pipeline, MCP enables:
- **Real-time pipeline monitoring** through Warp agents
- **Intelligent quality assessment** and recommendations 
- **Smart parameter tuning** based on garment and fabric types
- **Advanced debugging** and issue resolution
- **Route optimization** between SDXL and Gemini processing

## Quick Setup

1. **Install MCP dependencies:**
   ```bash
   python setup_mcp.py
   ```

2. **Configure Warp:**
   - Open Warp → Settings → AI → Manage MCP servers
   - Click "+ Add" and paste the configuration from `mcp_config.json`
   - Start the "photostudio-pipeline" server

3. **Test integration:**
   ```bash
   # Ask Warp agents questions like:
   # "What's the current status of the pipeline?"
   # "Analyze the quality of test_garment.jpg"
   # "Recommend parameters for processing a cotton shirt"
   ```

## Available MCP Tools

Our custom MCP server exposes these tools to Warp agents:

### 1. **analyze-garment-quality**
Performs comprehensive quality analysis on garment images using the pipeline's QA system.

**Usage:** *"Analyze the quality of this garment image: /path/to/image.jpg"*

**Returns:**
- Color accuracy (CIEDE2000 ΔE scores)
- Ghost-mannequin hollow quality assessment  
- Sharpness and background purity metrics
- Pass/fail status for quality gates
- Specific recommendations for improvements

### 2. **get-pipeline-status**
Provides real-time monitoring of all pipeline steps and processing queues.

**Usage:** *"What's the current status of the pipeline?"*

**Returns:**
- Active job count and queue length
- Health status of each processing step (0-6)
- Average processing times and success rates
- Auto-tuning system status and trends
- Alerts for any unhealthy components

### 3. **recommend-parameters**
Suggests optimal processing parameters based on garment type and fabric.

**Usage:** *"Recommend parameters for processing a silk dress"*

**Returns:**
- Optimal rendering route (SDXL vs Gemini)
- Processing strength recommendations
- Quality threshold settings
- Reasoning based on historical performance data

### 4. **debug-pipeline-step**
Provides detailed debugging assistance for specific pipeline steps.

**Usage:** *"Debug issues with Step 3 rendering"*

**Returns:**
- Step-specific common issues and solutions
- Log file locations for detailed analysis
- Performance bottleneck identification
- Memory and resource usage insights

### 5. **optimize-route-selection**
Analyzes garment characteristics to optimize Route A vs Route B selection.

**Usage:** *"Which rendering route should I use for this analysis file?"*

**Returns:**
- Route recommendation (A=SDXL, B=Gemini)
- Performance estimates (time/quality tradeoffs)
- Auto-switching trigger conditions
- Fabric-specific processing advice

## Integration with Existing Services

The MCP configuration also includes integration with external services:

### **GitHub Integration**
- Repository access for issue tracking and PR management
- Commit and branch information for development workflow
- Integration with pipeline CI/CD processes

### **Filesystem Access**  
- Direct access to pipeline input/output directories
- File analysis and batch processing capabilities
- Configuration file management

### **Notes Integration**
- Documentation and knowledge base access
- Quality assessment notes and historical data
- Best practice recommendations and templates

## Advanced Usage Examples

### Quality Assessment Workflow
```bash
# 1. Analyze a batch of images
"Analyze the quality of all images in ./input_batch/"

# 2. Get recommendations for failed images  
"What parameters should I adjust for images that failed hollow quality checks?"

# 3. Monitor improvement after parameter changes
"Show me the quality trends after the recent auto-tuning adjustments"
```

### Debugging Workflow
```bash
# 1. Identify problematic step
"Which pipeline step has the lowest success rate this week?"

# 2. Deep dive into specific issues
"Debug Step 3 rendering issues for job ID 12345"

# 3. Get optimization recommendations
"How can I reduce GPU memory usage in the SDXL route?"
```

### Parameter Optimization Workflow
```bash
# 1. Get fabric-specific recommendations
"What are the optimal settings for processing wool sweaters?"

# 2. Compare route performance
"Compare SDXL vs Gemini performance for complex textured fabrics"

# 3. Historical analysis
"Show me the auto-tuning changes that improved quality scores last month"
```

## Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
│   Warp Agent    │◄──►│  MCP Protocol    │◄──►│  PhotoStudio MCP    │
│                 │    │                  │    │     Server          │
└─────────────────┘    └──────────────────┘    └─────────────────────┘
                                                          │
                                                          ▼
                                               ┌─────────────────────┐
                                               │  Pipeline Core      │
                                               │  - Step 0-6         │
                                               │  - QA System        │
                                               │  - Auto-tuning      │
                                               │  - Telemetry        │
                                               └─────────────────────┘
```

## Configuration Files

- **`mcp_config.json`** - Warp MCP server configuration
- **`mcp_server/photostudio_mcp.py`** - Custom MCP server implementation  
- **`mcp_server/requirements.txt`** - MCP-specific dependencies
- **`setup_mcp.py`** - Automated setup and configuration script
- **`.env.mcp`** - Environment variables for MCP servers (created by setup)

## Security Considerations

1. **API Keys:** Store sensitive tokens in environment variables, not configuration files
2. **File Access:** The filesystem MCP server is restricted to the project directory
3. **Network Access:** External MCP servers (GitHub, etc.) require appropriate authentication
4. **Logging:** MCP server logs may contain sensitive information - review before sharing

## Troubleshooting

### Common Issues

**MCP Server Won't Start:**
- Check Python path and virtual environment
- Verify all dependencies are installed
- Review MCP server logs in Warp settings

**Tools Not Available in Warp:**
- Ensure MCP server is running (green status in Warp)
- Check server configuration syntax
- Restart Warp after configuration changes

**Permission Errors:**
- Verify working directory permissions
- Check environment variable access
- Validate Docker permissions for containerized servers

### Debug Commands

```bash
# Test MCP server directly
python mcp_server/photostudio_mcp.py

# Check dependencies  
pip list | grep mcp

# Validate configuration
python -c "import json; print(json.load(open('mcp_config.json')))"

# Review logs
# Check Warp → Settings → AI → MCP Servers → [server] → View Logs
```

## Future Enhancements

Potential expansions for the MCP integration:

1. **Real-time Notifications:** Push alerts for quality failures or processing bottlenecks
2. **Batch Operations:** Mass processing and analysis capabilities  
3. **ML Model Management:** Dynamic model loading and parameter optimization
4. **Cloud Integration:** AWS/GCP storage and processing extensions
5. **Workflow Automation:** Automated pipeline orchestration based on agent recommendations

## Contributing

To extend the MCP server with new tools:

1. Add tool definition to `handle_list_tools()` in `photostudio_mcp.py`
2. Implement the tool function following the existing patterns
3. Add the tool call to `handle_call_tool()` dispatcher
4. Update this documentation with usage examples
5. Test thoroughly with various Warp agent interactions

## Support

For issues with MCP integration:
- Check the troubleshooting section above
- Review Warp MCP documentation: https://docs.warp.dev/knowledge-and-collaboration/mcp
- Examine MCP server logs for detailed error messages
- Ensure all pipeline dependencies are properly installed
