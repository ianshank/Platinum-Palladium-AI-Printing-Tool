---
description: Setup Husky Git Hooks
---

# Setup Husky Git Hooks

Automate code quality checks with pre-commit and pre-push hooks.

## Steps

1. **Install Husky and lint-staged**

    ```bash
    npm install --save-dev husky lint-staged
    ```

2. **Initialize Husky**

    ```bash
    npx husky init
    ```

3. **Configure lint-staged**
    Add to `package.json`:

    ```json
    {
      "lint-staged": {
        "*.{ts,tsx}": [
          "eslint --fix",
          "prettier --write"
        ],
        "*.{json,md}": [
          "prettier --write"
        ]
      }
    }
    ```

4. **Add Pre-commit Hook**

    ```bash
    echo "npx lint-staged" > .husky/pre-commit
    ```

5. **Add Pre-push Hook**

    ```bash
    echo "npm test" > .husky/pre-push
    ```

    *Note: On Windows PowerShell, use `Set-Content` or verify the file content manually.*

6. **Install Commitlint (Optional but Recommended)**

    ```bash
    npm install --save-dev @commitlint/cli @commitlint/config-conventional
    echo "module.exports = { extends: ['@commitlint/config-conventional'] };" > commitlint.config.js
    echo "npx --no -- commitlint --edit \$1" > .husky/commit-msg
    ```
