#!/bin/bash

nvcc --extended-lambda -ptx reduceCUDA.cu -o reduceCUDA.ptx
