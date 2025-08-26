# QuASAr

This repository contains experimental utilities for the QuASAr project.  The
`quasar_convert` package optionally provides a native C++ backend for faster
conversion routines.

## Installation

Building from source requires a C++17 compiler and a recent version of Python.
The native extension is built automatically when installing the package:

```bash
pip install .
```

If a compiler is not available the build will fall back to a pure Python stub
implementation with reduced performance but identical APIs.

