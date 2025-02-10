# PIM-common
Common functions &amp; macros to be shared by all projects

## Organization

|Directory        | Purpose                                             |
|-----------------|-----------------------------------------------------|
|common/include   | Macros and function definitions for common code     |
|dpu/include      | Macros and function definitions for DPU-only code   |
|host/include     | Macros and function definitions for host-only code  |

## How to add this to your project

Add this to your existing PIM project with:
```
git submodule add -- https://github.com/UBC-ECE-Sasha/PIM-common.git
```
The 'http' URL is preferred over the 'git' URL in case someone wants to clone your project without having git access (i.e. unauthenticated) to these projects.
