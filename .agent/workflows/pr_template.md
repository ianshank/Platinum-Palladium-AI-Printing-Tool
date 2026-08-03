---
description: Create GitHub Pull Request Template
---

# Create GitHub PR Template

Standardize pull request descriptions for better code reviews.

## Steps

1. **Create Template Directory**

    ```bash
    mkdir -p .github
    ```

2. **Create Template File**
    Create `.github/pull_request_template.md` with the following content:

    ```markdown
    ## Description
    Briefly describe what this PR does.

    ## Type of Change
    - [ ] Bug fix
    - [ ] New feature
    - [ ] Breaking change
    - [ ] Documentation update

    ## Testing
    - [ ] I have tested these changes locally
    - [ ] I have added/updated tests
    - [ ] All tests pass

    ## Screenshots (if applicable)

    ## Checklist
    - [ ] My code follows the project's style guidelines
    - [ ] I have performed a self-review
    - [ ] I have commented complex code
    - [ ] I have updated documentation
    - [ ] No new warnings generated

    ## Related Issues
    Closes #
    ```

3. **Commit and Push**

    ```bash
    git add .github/pull_request_template.md
    git commit -m "chore: add PR template"
    git push
    ```

4. **Test It**
    Create a new PR to verify the template auto-populates.
