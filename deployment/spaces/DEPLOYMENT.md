# Deploying to HuggingFace Spaces

## Step-by-Step Deployment Guide

### 1. Create HuggingFace Account

Sign up at [huggingface.co](https://huggingface.co/join) if you don't have an account.

### 2. Create New Space

1. Go to [https://huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Fill in details:
   - **Space name**: `code-generation`
   - **License**: MIT
   - **SDK**: Gradio
   - **Hardware**: CPU basic (free) or T4 small (for faster inference)

### 3. Prepare Files

From the `deployment/spaces/` directory:

```
spaces/
├── app.py           # Gradio app (required)
├── requirements.txt # Dependencies (required)
├── README.md        # Space description (required)
└── best_model/      # LoRA adapter (required)
```

### 4. Upload Model Adapter

Copy your fine-tuned model adapter:

```bash
# From project root
cp -r results/models/qlora-exaone-2.4b/best_model deployment/spaces/
```

### 5. Upload to Space

#### Option A: Using Git (Recommended)

```bash
cd deployment/spaces

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/code-generation
cd code-generation

# Copy files
cp ../app.py .
cp ../requirements.txt .
cp ../README.md .
cp -r ../best_model .

# Add and commit
git add .
git commit -m "Initial deployment"

# Push to HuggingFace
git push
```

#### Option B: Using Web Interface

1. Go to your Space page
2. Click "Files and versions"
3. Upload files:
   - `app.py`
   - `requirements.txt`
   - `README.md`
   - `best_model/` (all files in the directory)

### 6. Wait for Build

HuggingFace will automatically:
1. Install dependencies from `requirements.txt`
2. Download base model (EXAONE-3.5-2.4B-Instruct)
3. Load LoRA adapter
4. Launch Gradio app

Build time: ~5-10 minutes

### 7. Test Your Space

Once deployed, your Space will be available at:
```
https://huggingface.co/spaces/YOUR_USERNAME/code-generation
```

Test it by:
1. Entering a code prompt
2. Clicking "Generate Code"
3. Verifying output quality

## Troubleshooting

### Build Failed

Check build logs:
1. Go to your Space
2. Click "Community" tab
3. Look for error messages

Common issues:
- **Out of memory**: Upgrade to paid GPU tier
- **Missing files**: Ensure all files uploaded
- **Import errors**: Check `requirements.txt`

### Slow Performance

Free CPU tier is slow. Upgrade hardware:
1. Go to Space Settings
2. Select "Hardware"
3. Choose "T4 small" ($0.60/hour) or "T4 medium"

### Model Not Loading

Ensure `best_model/` contains:
- `adapter_config.json`
- `adapter_model.safetensors`
- All tokenizer files

## Configuration Options

### Hardware Tiers

| Tier | vCPU | RAM | GPU | Cost | Speed |
|------|------|-----|-----|------|-------|
| CPU basic | 2 | 16 GB | - | **FREE** | Slow |
| CPU upgrade | 8 | 32 GB | - | $0.03/hr | Better |
| T4 small | 4 | 16 GB | 16 GB | $0.60/hr | Fast |
| T4 medium | 8 | 32 GB | 16 GB | $1.05/hr | Fastest |

### Visibility

- **Public**: Anyone can view and use
- **Private**: Only you can access

Set in Space Settings.

### Sleep Mode

Free Spaces sleep after 48 hours of inactivity. Paid hardware stays active.

## Customization

### Update README

Edit `README.md` to customize:
- Title and description
- Example prompts
- Performance metrics
- Links to your GitHub

### Modify UI

Edit `app.py` to:
- Change theme: `theme=gr.themes.Soft()`
- Add more examples
- Adjust parameter ranges
- Customize layout

### Add Features

Enhance the app:
- Multiple model selection
- Code syntax highlighting
- Export to file
- Share generated code

## Monitoring

### Usage Stats

View in Space Settings:
- Total runs
- Unique users
- Average response time

### Logs

Check logs for errors:
1. Go to Space
2. Click "Logs" tab
3. View real-time output

## Cost Management

### Free Tier Limits

- CPU basic: Unlimited (with sleep)
- Storage: 50 GB
- Bandwidth: Unlimited

### Paid Usage

Only pay when Space is active:
- T4 small: ~$432/month (continuous)
- T4 medium: ~$756/month (continuous)

**Pro Tip**: Use free CPU tier for demo, upgrade temporarily for presentations.

## Updating Your Space

### Update Code

```bash
cd code-generation

# Edit files
vim app.py

# Commit and push
git add app.py
git commit -m "Update generation parameters"
git push
```

Space will automatically rebuild.

### Update Model

```bash
# Copy new adapter
cp -r ../../results/models/qlora-exaone-2.4b/best_model .

# Push update
git add best_model/
git commit -m "Update model adapter"
git push
```

## Best Practices

1. **Test locally first**: Run `gradio app.py` before deploying
2. **Use small model**: Keep under 10 GB for free tier
3. **Add examples**: Help users understand your model
4. **Monitor logs**: Check for errors after deployment
5. **Document well**: Good README increases usage

## Alternative: Gradio Share

For temporary sharing (no HuggingFace account needed):

```python
# In app.py
demo.launch(share=True)
```

This creates a temporary public URL (valid for 72 hours).

## Resources

- [HuggingFace Spaces Docs](https://huggingface.co/docs/hub/spaces)
- [Gradio Documentation](https://gradio.app/docs/)
- [PEFT Documentation](https://huggingface.co/docs/peft/)

## Support

For issues:
1. Check [HuggingFace Forums](https://discuss.huggingface.co/)
2. Open issue in [GitHub repo](https://github.com/kimddong23/llm-finetuning-pipeline/issues)
3. Contact via Space comments

---

**Ready to deploy?** Follow the steps above and your code generation model will be live in minutes! 🚀
