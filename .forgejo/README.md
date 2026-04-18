# Forgejo CI Configuration

This directory contains Forgejo Actions workflows.

## Mirror to GitHub Workflow

The `mirror-github.yml` workflow automatically mirrors all pushes from this
Forgejo instance to the GitHub repository. It runs on every push to any branch
or tag.

### Required Secrets

Configure the following secrets in the Forgejo repository settings:

- `USER_GITHUB`: GitHub username or organization name (e.g., `ethank5149`)
- `PAT_GITHUB`: GitHub Personal Access Token with `repo` scope to push to the repository

### How it works

1. On push, the workflow checks out the repository with full history.
2. Adds a remote pointing to the GitHub repository using the PAT.
3. Executes `git push --mirror --force` to synchronize all refs (branches, tags,
   and other refs) exactly to GitHub.

The mirror is unidirectional: Forgejo is the source of truth. All pushes should
be made to Forgejo; the workflow automatically propagates them to GitHub.