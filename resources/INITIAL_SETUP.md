# Initial Setup

## Create a new repository using this template

1. Navigate to your **Repositories** tab on GitHub (https://github.com/$USERNAME?tab=repositories).
1. Click on **New** in the upper right hand corner.
1. In the drop-down menu under **Repository template**, select `tamaskis/rust-github-template`.
1. Fill out the **Repository name**.
1. Initially, set the repository to **Private**.
1. Click **Create repository**.

## Setting up branch protection rules

1. Navigate to your repository on GitHub (https://github.com/$USERNAME/$REPOSITORY).
1. Click on the **Settings** tab.
1. Make sure you are on the **General page**.
1. Scroll down to the **Pull Requests** section and:
    1. Disable **Allow merge commits**.
    1. In the drop-down menu under **Default commit message** below **Allow squash merging**, select **Pull request title**.
    1. Disable **Allow rebase merging**.
    1. Enable **Automatically delete head branches**.
1. On the left hand side, under **Code and automation**, click on **Branches**.
1. Click on **Add classic branch protection rule**.
1. Set **Branch name pattern** to `main`.
1. Enable **Require a pull request before merging**.
    1. Enable **Dismiss stale pull request approvals when new commits are pushed**.
    1. Enable **Require approval of the most recent reviewable push**.
1. Enable **Require status checks to pass before merging**.
    1. Enable **Require branches to be up to date before merging**.
1. Enable **Require conversation resolution before merging**.
1. Enable **Require linear history**.
1. Enable **Lock branch**.

## Cloning the repo

1. Navigate to your repository on GitHub (https://github.com/$USERNAME/$REPO_NAME).
2. Click the green **Code** drop down menu, and copy the web URL from the **HTTPS** tab under **Clone**.
3. Open a terminal.
4. Navigate to the directory storing all the Rust projects.

    ```
    cd Documents/projects/rust
    ```

5. Clone the repo, copy/pasting the web URL copied in step 8.

    ```
    git clone https://github.com/tamaskis/$REPO_NAME.git
    ```

## Initial crate setup

1. Open `Documents/projects/rust/$REPO_NAME` in a new VS Code window.
1. Create a branch for the initial crate setup.

    ```
    git checkout -b initial-crate-setup
    git push --set-upstream origin initial-crate-setup
    ```

1. Delete the `resources/` folder.
1. Perform a global search/replace (`Ctrl + Shift + F`) for `TODO_YEAR`, replacing all instances with the actual year.
1. Perform a global search/replace (`Ctrl + Shift + F`) for `todo-crate-name`, replacing all instances with the actual crate name.
1. Perform a global search/replace (`Ctrl + Shift + F`) for `TODO_CRATE_NAME_WITH_UNDERSCORES`, replacing all instances with the actual crate name but with dashes replaced with underscores.
1. Perform a global search/replace (`Ctrl + Shift + F`) for `TODO_CRATE_NAME_WITH_DASHES`, replacing all instances with the actual crate name, including dashes (i.e. `-`).
1. Perform a global search/replace (`Ctrl + Shift + F`) for `TODO_REPO`, replacing all instances with the actual repository name.
1. Perform a global search/replace (`Ctrl + Shift + F`) for `TODO_DESCRIPTION`, replacing all instances with the actual description.
1. Stage, commit, and push your changes.

    ```
    git add .
    git commit -m "initial crate setup"
    git push
    ```
1. Navigate to your repository on GitHub (https://github.com/$USERNAME/$REPO_NAME).
1. There should be a yellow box with the message "**initial-crate-setup** had recent pushes X seconds ago". In this yellow box, click on the green **Compare & pull request** button.
1. On the next page, click on **assign yourself** under **Assignees** on the right hand side, then click on the green **Create pull request** button at the bottom of the page.
1. On the next page, click on the green **Merge pull request button** and then click on the green **Confirm merge** button.
1. Back in VS Code, checkout the `main` branch, pull the latest version, and delete the `initial-crate-setup` branch.

    ```
    git checkout main
    git pull origin main
    git branch -D initial-crate-setup
    ```

## Final GitHub setup

1. Navigate to your repository on GitHub (https://github.com/$USERNAME/$REPO_NAME).
1. On the right hand side, click on the gear icon next to **About**.
1. Fill out the **Description** box with whatever you set as your crate description in the [Initial crate setup](#initial-crate-setup) section, then click on the green **Save changes** button.
1. Click on the **Settings** tab.
1. Make sure you are on the **General page**.
1. Scroll down to the bottom of the page, and under the **Danger Zone** section click on the **Change visibility** button. Then, click on **Change to public** and finally follow the remaining prompts.
1. After setting the repository to public, GitHub should take you back to the **Settings** tab. On the left hand side, under **Code and automation**, click on **Branches**.
1. Click on the **Edit** button for the `main` branch protection rule.
1. In the **Protect matching branches** sections, enable **Require review from Code Owners** (under **Require a pull request before merging**).
1. Scroll to the bottom of the page and click on the green **Save changes** button.