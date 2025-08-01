# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from datetime import timezone, datetime

__versioninfo__ = (22, 0, 0)
__version__ = (
    ".".join(str(v) for v in __versioninfo__)
    + "dev"
    + datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M")
)
