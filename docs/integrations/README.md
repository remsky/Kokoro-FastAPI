# Integrations

*Last updated: 2026-08-11*

Compatible with anything that works with the OpenAI TTS endpoint. If something's missing or broken, open an issue or PR.

- [OpenAI](openai.md)
- [OpenWebUI](openwebui.md)
- [SillyTavern](sillytavern.md)

Deployment guides (DigitalOcean, Kubernetes) are in [deployment/](../deployment/).

## Adding an integration

PRs welcome. Guidelines:

- A page in `docs/integrations/`, include a `Last updated` line for functional changes.
- Match styling badge inline with the others in the root README if appropriate.
- Flag anything requiring an account, key, code; paid tier vs free, etc
- Keep it simple, and focused on setup, deploy, connect, any gotchas.
- Screenshots in `docs/assets/`, organized if need be. 
- Use `:latest` for image tags for simplicity, or pin if there's a specific version req. 
- Commands should be copy-pasteable when possible e.g. backtick wrapped
