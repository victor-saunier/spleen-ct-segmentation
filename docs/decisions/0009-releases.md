# 0009 - Publishing releases

- Status: accepted
- Date: 2026-11-23

## Context

The Docker image (decision record 0008) is built and tested in CI on every pull request, but
using it still means cloning the repository and building it. A release should give a fixed,
named version that anyone can pull and run, published only when the owner decides, and only
from a commit that passed every check.

## Decisions

| Topic | Choice | Options rejected |
|---|---|---|
| Trigger | Pushing a `v*` git tag (semantic version, such as `v1.0.0`) | Every push to main: publishes versions nobody chose. Manual uploads: not reproducible. |
| Where | A `publish` job in the CI workflow, after the `checks` job passes on the tagged commit | A separate workflow: repeats the setup and could publish without the tests. |
| Rights | `packages: write` and `contents: write` in the `publish` job only | Write rights for the whole workflow: pull request runs would hold them too. |
| Registry | GitHub container registry (`ghcr.io`), under the repository's name | Docker Hub: a second account and a token stored as a secret. |
| Login | `docker login` with the workflow's own short-lived token | `docker/login-action`: one more third-party action to trust and pin. |
| Image name | The version without the `v` (`:1.0.0`); the release notes also give the digest | `:latest`: a name that moves, the problem digests solve (decision record 0008). |
| Image built | Rebuilt from the tagged commit in the `publish` job | Passing the tested image between jobs: a 450 MB transfer. The locked libraries and digest-pinned base images give the same contents, but the file is not literally the one tested. |
| Model weights | No model file is released; the image has none (as in 0008) | Releasing `best.pt` or the ONNX file. The weights were trained on CC BY-SA 4.0 data; whether they count as adapted material, which would require the same licence and attribution, is not settled. The conservative choice is not to publish them. If they are published later, then under CC BY-SA 4.0 with attribution, not under the code's MIT licence. |
| Pre-releases | Versions `0.x` are marked as pre-releases | All versions final: the first tag only tests the pipeline. |
| Not done | Image signing, a software bill of materials, other CPU architectures | Useful in production; not needed to show this pipeline, listed as follow-ups. |

## Consequences

- A release is one command by the owner: `git tag v1.2.3 <commit on main>` then
  `git push origin v1.2.3`. A tag pushed by mistake publishes, so tags are pushed by hand only.
- Users of the image must bring their own trained model folder: without released weights, the
  image alone cannot predict. This is the main limit of the release.
- The repository is private until the project is finished; who can pull the image depends on
  the package's visibility on GitHub, checked after the first release.
