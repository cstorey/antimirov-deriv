#!/bin/sh
exec sed -ne "s/^test \([^ ]*\)_0*\([0-9]*\) .*bench: *\([0-9][0-9,]*\) \([^ ]*\) .*/\1;\2;\3;\4/p"  "$@" | tr -d , | tr \; ,
