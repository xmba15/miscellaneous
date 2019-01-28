#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

g++ -std=c++11 -o test testSerialization.cpp -lboost_system -lboost_serialization
