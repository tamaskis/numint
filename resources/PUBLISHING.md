# Publishing

## Prequisites

1. Set up your `crates.io` API token: https://doc.rust-lang.org/cargo/reference/publishing.html.
    * Scopes: `publish-new` and `publish-update`.

## GitHub Release

These instructions assume you are currently working on a branch off of `main`.

1. Update the version in `Cargo.toml` to `x.y.z`.
1. Add a changelog entry corresponding to `x.y.z` in the `CHANGELOG.md`.
1. Merge your branch with package version `x.y.z` to the `main` branch on GitHub via a pull request.
1. Pull the `main` branch in your local repository.

    ```
    git pull origin main
    ```

1. Create a new tag and add a message.

    ```
    git tag -a x.y.z -m "Release version x.y.z"
    ```

1. Push the tag to the remote repository.

    ```
    git push origin x.y.z
    ```

1. Release on GitHub.
    1. In your repository on GitHub, click on **Releases** on the right hand side.
    1. Click on the **Draft a new release** button.
    1. Under the **Choose a tag** drop-down menu, select `x.y.z`.
    1. Set **Release title** to `Release version x.y.z`.

## Publish to `crates.io`

1. Publish to `crates.io`.

    ```
    cargo publish --all-features
    ```