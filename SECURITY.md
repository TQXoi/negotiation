# Security and publication checklist

Never commit API keys, access tokens, private endpoints, model checkpoints, raw
provider responses, or private datasets.

## Configuration

- Read credentials from environment variables.
- Keep local values in an ignored `.env.local` if necessary.
- Commit only `.env.example` files with empty values.
- Do not put credentials directly into shell scripts, JSON configs, notebooks,
  reports, presentations, or copied terminal logs.

## Before every push

From the repository root:

```bash
git status --short
git diff --cached --check
git grep -n -I -E 'sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9_]{20,}|AIza[0-9A-Za-z_-]{20,}|AKIA[0-9A-Z]{16}' -- .
find . -type f -size +25M -not -path './.git/*' -print
```

Also inspect staged environment/config files manually:

```bash
git diff --cached --name-only | grep -E '(^|/)(\.env|config|credential|secret|token|key)' || true
```

If a real secret is ever committed, deleting the file in a later commit is not
enough. Revoke/rotate the secret immediately and rewrite the repository history
before publishing.
